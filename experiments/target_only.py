"""Train ChromaGRT on one frozen ESANN target-only fold with RT/duration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromagrt-root', type=Path, required=True)
    parser.add_argument('--fold-dir', type=Path, required=True)
    parser.add_argument('--molecular-descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=['random', 'bm', 'cc'], default='random')
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--max-epochs', type=int, default=250)
    args, architecture_args = parser.parse_known_args()
    base = args.chromagrt_root.resolve()
    pin = json.loads((Path(__file__).resolve().parents[1] / 'chromagrt-base.json').read_text())
    actual = subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin['commit']:
        parser.error(f"ChromaGRT must be checked out at {pin['commit']}; found {actual}")
    if args.max_epochs < 1:
        parser.error('--max-epochs must be positive')
    sys.path.insert(0, str(base))
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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(config['seed'], workers=True)
    raw, scaled, audit = scaled_inputs(parts, config, args.molecular_descriptors)
    scaled, audit['rt_normalization'] = add_duration_inputs(raw, scaled, config)
    (output / 'preprocessing_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    bundle = train_chromagrt(scaled['train'], scaled['valid'], config,
                             str(output) + '/', using_moldescs=True, save_model=True)
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
