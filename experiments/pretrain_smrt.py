"""Pretrain on the frozen SMRT split and export the best validation checkpoint."""

import argparse
import json
from pathlib import Path

from target_only import load_chromagrt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromagrt-root', type=Path, required=True)
    parser.add_argument('--smrt-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--max-epochs', type=int, default=250)
    args, architecture_args = parser.parse_known_args()
    commit = load_chromagrt(args.chromagrt_root)

    import numpy as np
    import pandas as pd
    from lightning import pytorch as pl
    from architecture_backend import train_chromagrt, predict_chromagrt
    from model_architecture import Architecture, add_arguments
    from preprocessing import scaled_inputs
    from program_duration import MODE, DEFINITION, add_duration_inputs
    from src.training.functions.splitted_sets_functions import FOLD_IMPUTATION_COLUMNS
    from src.training.model_backends.chromagrt_dataset import get_condition_columns
    from src.training.model_backends.chromagrt_defaults import DEFAULT_CHROMAGRT_CONFIG
    from transfer_core import digest, input_encoding

    architecture_parser = argparse.ArgumentParser(add_help=False)
    add_arguments(architecture_parser)
    architecture = Architecture.from_args(architecture_parser.parse_args(architecture_args))
    if args.max_epochs < 1:
        parser.error('--max-epochs must be positive')
    config = dict(DEFAULT_CHROMAGRT_CONFIG, **architecture.as_dict(),
                  accelerator=args.accelerator, condition_normalization='deterministic',
                  tanaka_block_dropout=0., rt_normalization=MODE, rt_duration_definition=DEFINITION,
                  max_epochs=args.max_epochs, seed=42)
    def read(name):
        return pd.read_csv(args.smrt_dir / name, sep='\t', dtype={'dir_id': str})

    parts = {role: read(f'{role}_data.tsv') for role in ('train', 'valid')}
    for role, frame in parts.items():
        if frame.empty or not frame.molecule_id.str.startswith('0186_').all():
            raise ValueError(f'Expected SMRT observations in {role}')
    if set(parts['train']['inchi.std']) & set(parts['valid']['inchi.std']):
        raise ValueError('SMRT train and validation identities overlap')
    parts['test'] = parts['valid'].assign(split='test')
    unknown_fields = []
    for column in FOLD_IMPUTATION_COLUMNS:
        if parts['train'][column].isna().all():
            if column != 'column.temperature':
                raise ValueError(f'Unresolved SMRT method metadata: {column}')
            unknown_fields.append(column)
            for part in parts.values():
                part[column] = part[column].fillna(0.)
    pl.seed_everything(config['seed'], workers=True)
    raw, scaled, audit = scaled_inputs(parts, config, args.smrt_dir / 'molecular_descriptors.tsv')
    scaled, audit['rt_normalization'] = add_duration_inputs(raw, scaled, config)
    audit['condition_columns'] = get_condition_columns(
        scaled['train'], using_moldescs=True, include_tanaka=True)
    audit['unknown_method_fields_neutral_zero'] = unknown_fields
    audit['input_encoding'] = input_encoding(raw['train'], config, audit)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'preprocessing_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    bundle = train_chromagrt(scaled['train'], scaled['valid'], config,
                             str(output) + '/', using_moldescs=True)
    predictions = predict_chromagrt(bundle, scaled['valid'])
    rows = raw['valid'][['molecule_id', 'inchi.std', 'rt']].copy()
    if len(rows) != len(predictions) or not np.isfinite(predictions).all():
        raise RuntimeError('Missing or invalid validation predictions')
    rows['pred_rt'] = predictions
    rows.to_csv(output / 'Validation.tsv', sep='\t', index=False)
    logged = pd.read_csv(output / 'lightning_logs/version_0/metrics.csv').dropna(subset=['val_mae'])
    best = logged.loc[logged.val_mae.idxmin()]
    provenance = json.loads((args.smrt_dir / 'preparation.json').read_text())
    manifest = dict(
        format='chromagrt-smrt-v1', source_dataset='SMRT_0186', complete=True,
        smoke=(len(parts['train']), len(parts['valid'])) != (70184, 7799),
        architecture=architecture.as_dict(), condition_columns=bundle['condition_columns'],
        input_encoding=audit['input_encoding'], rt_normalization=MODE,
        target_mean=bundle['model'].target_mean, target_std=bundle['model'].target_std,
        scaffold_filtering=provenance['scaffold_filtering'],
        missing_metadata_policy=provenance['missing_metadata_policy'],
        weights_sha256=digest(output / 'model.pt'), chromagrt_commit=commit,
        best_epoch_zero_based=int(best.epoch), best_validation_MAE_seconds=float(best.val_mae),
        actual_training_rows=len(parts['train']), actual_validation_rows=len(parts['valid']),
        config=bundle['config'],
    )
    (output / 'pretrain_checkpoint.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
