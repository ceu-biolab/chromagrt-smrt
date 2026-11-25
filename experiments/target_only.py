"""Train ChromaGRT on one frozen ESANN fold with RT/duration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def load_chromagrt(root):
    base = root.resolve()
    pin = json.loads((Path(__file__).resolve().parents[1] / 'chromagrt-base.json').read_text())
    actual = subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin['commit']:
        raise ValueError(f"ChromaGRT must be checked out at {pin['commit']}; found {actual}")
    sys.path.insert(0, str(base))
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromagrt-root', type=Path, required=True)
    parser.add_argument('--fold-dir', type=Path, required=True)
    parser.add_argument('--molecular-descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=['random', 'bm', 'cc'], default='random')
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--max-epochs', type=int, default=250)
    parser.add_argument('--strategy', choices=['target-only', 'transfer', 'joint'], default='target-only')
    parser.add_argument('--smrt-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--fraction', choices=['0.25', '0.5', '1'], default='1')
    parser.add_argument('--fold', type=int, choices=range(10))
    parser.add_argument('--matched-updates', action='store_true')
    args, architecture_args = parser.parse_known_args()
    if (args.strategy == 'transfer') != (args.checkpoint is not None):
        parser.error('Use --checkpoint exactly when selecting transfer')
    if args.strategy == 'transfer' and args.scenario == 'bm':
        parser.error('BM transfer was outside the paper protocol')
    if (args.strategy == 'joint') != (args.smrt_dir is not None):
        parser.error('Use --smrt-dir exactly when selecting joint')
    if args.fraction != '1' and (args.scenario != 'random' or args.fold is None):
        parser.error('A reduced fraction requires random and an explicit --fold')
    if args.matched_updates and (args.strategy, args.fraction) != ('target-only', '0.25'):
        parser.error('The matched-updates protocol uses target-only at fraction 0.25')
    actual = load_chromagrt(args.chromagrt_root)
    if args.max_epochs < 1:
        parser.error('--max-epochs must be positive')
    from model_architecture import Architecture, add_arguments
    from program_duration import DEFINITION, MODE, add_duration_inputs

    architecture_parser = argparse.ArgumentParser(add_help=False)
    add_arguments(architecture_parser)
    architecture = Architecture.from_args(architecture_parser.parse_args(architecture_args))
    import numpy as np
    import pandas as pd
    from lightning import pytorch as pl
    from architecture_backend import predict_chromagrt, train_chromagrt
    from preprocessing import scaled_inputs
    from partitions import RepeatedShuffle, append_smrt, target_subset
    from src.training.functions.splitted_sets_functions import get_res_table
    from src.training.model_backends.chromagrt_defaults import DEFAULT_CHROMAGRT_CONFIG

    config = dict(
        DEFAULT_CHROMAGRT_CONFIG, **architecture.as_dict(),
        accelerator=args.accelerator, max_epochs=args.max_epochs, seed=42,
        condition_normalization='deterministic' if args.scenario == 'cc' else 'standard',
        tanaka_block_dropout=0.5 if args.scenario == 'cc' else 0.0,
        rt_normalization=MODE, rt_duration_definition=DEFINITION, rt_loss='MAE_seconds',
        rt_output_upper_bound=None,
        rt_duration_input='program.duration; train-CC standardization or deterministic minutes',
    )
    parts = {
        role: pd.read_csv(args.fold_dir / name, sep='\t', dtype={'dir_id': str})
        for role, name in [('train', 'train_data.tsv'), ('valid', 'val_data.tsv'), ('test', 'test_data.tsv')]
    }
    if any(frame.molecule_id.str.startswith('0186_').any() for frame in parts.values()):
        raise ValueError('The target fold must contain only non-SMRT observations')
    if args.fraction != '1':
        parts['train'] = target_subset(parts['train'], args.fold, args.fraction)
    n_target = len(parts['train'])
    descriptor_path = args.molecular_descriptors
    if args.strategy == 'joint':
        smrt = pd.read_csv(args.smrt_dir / 'train_data.tsv', sep='\t', dtype={'dir_id': str})
        parts['train'] = append_smrt(parts, smrt, args.scenario)
        descriptors = pd.concat([
            pd.read_csv(path, sep='\t') for path in
            (descriptor_path, args.smrt_dir / 'molecular_descriptors.tsv')
        ], ignore_index=True).drop_duplicates()
        if descriptors.inchi.duplicated().any():
            raise ValueError('Conflicting descriptor rows for one InChI')
    sampler = None
    if args.matched_updates:
        from math import ceil

        budgets = json.loads((Path(__file__).resolve().parent / 'matched-budgets.json').read_text())
        budget = budgets[str(args.fold)]
        if (architecture.as_dict() != dict(hidden_dim=256, num_heads=8, num_layers=6, ffn_dim=512)
                or budget['target_rows'] != n_target or budget['batch_size'] != config['batch_size']):
            raise ValueError('Matched budget requires the published architecture, subset and batch size')
        if ceil(budget['draws'] / config['batch_size']) != budget['steps_per_validation']:
            raise ValueError('Inconsistent matched budget')
        config['max_epochs'] = budget['max_epochs']
        sampler = RepeatedShuffle(n_target, budget['draws'])

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.strategy == 'joint':
        descriptor_path = output / 'molecular_descriptors.tsv'
        descriptors.to_csv(descriptor_path, sep='\t', index=False)
    pl.seed_everything(config['seed'], workers=True)
    raw, scaled, audit = scaled_inputs(parts, config, descriptor_path)
    scaled, audit['rt_normalization'] = add_duration_inputs(raw, scaled, config)
    audit.update(strategy=args.strategy, fraction=args.fraction, fold=args.fold,
                 n_target_train=n_target, n_smrt_train=len(parts['train']) - n_target)
    if args.matched_updates:
        audit['matched_budget'] = budget
    (output / 'preprocessing_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    bundle = train_chromagrt(scaled['train'], scaled['valid'], config,
                             str(output) + '/', using_moldescs=True, save_model=True,
                             training_sampler=sampler, pretrained_checkpoint=args.checkpoint)
    if args.matched_updates:
        logged = pd.read_csv(output / 'lightning_logs/version_0/metrics.csv').dropna(subset=['val_mae'])
        steps = (logged.step + 1).tolist()
        interval = budget['steps_per_validation']
        if steps != list(range(interval, int(bundle['trainer'].global_step) + 1, interval)):
            raise RuntimeError('Validation cadence differs from the Joint schedule')
        observed = int(bundle['trainer'].global_step)
        if not 0 < observed <= budget['max_steps']:
            raise RuntimeError('Optimizer updates exceeded the planned cap')
        (output / 'optimizer_updates.json').write_text(json.dumps({
            'observed': observed, 'cap': budget['max_steps'],
            'reference_run_observed': budget['observed_steps'],
        }, indent=2) + '\n')
    predictions = predict_chromagrt(bundle, scaled['test'])
    if len(predictions) != len(parts['test']) or not np.isfinite(predictions).all():
        raise RuntimeError('Missing or invalid test predictions')
    results = get_res_table(raw['test'], predictions, str(output) + '/', using_moldescs=True)
    results.to_csv(output / 'Results.tsv', sep='\t', index=False)
    (output / 'model_config.json').write_text(json.dumps({
        'chromagrt_commit': actual,
        'scenario': args.scenario,
        'condition_columns': bundle['condition_columns'],
        'config': bundle['config'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
