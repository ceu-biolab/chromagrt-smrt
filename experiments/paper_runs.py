"""Canonical prediction directory names and runners for the paper experiments."""

import argparse
from pathlib import Path
import subprocess
import sys

ARCHITECTURES = {
    96: (3, 6, 192),
    128: (4, 6, 256),
    256: (8, 6, 512),
    512: (16, 8, 1024),
}


def cases():
    for width in ARCHITECTURES:
        for strategy in ('target-only', 'transfer', 'joint'):
            yield 'random', width, '1', strategy
    for fraction in ('0.25', '0.5'):
        for strategy in ('target-only', 'transfer', 'joint'):
            yield 'random', 256, fraction, strategy
    yield 'random', 256, '0.25', 'matched-updates'
    for scenario, strategies in [('bm', ('target-only', 'joint')),
                                 ('cc', ('target-only', 'transfer', 'joint'))]:
        for strategy in strategies:
            yield scenario, 512, '1', strategy


def name(case):
    scenario, width, fraction, strategy = case
    return f'{scenario}/h{width}/fraction_{fraction}/{strategy}'


def published_folds(case):
    return range(4) if case == ('random', 128, '1', 'joint') else range(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=[name(case) for case in cases()])
    parser.add_argument('--chromagrt-root', required=True, type=Path)
    parser.add_argument('--target-dir', required=True, type=Path)
    parser.add_argument('--smrt-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(10))
    args = parser.parse_args()
    case = next(case for case in cases() if name(case) == args.case)
    scenario, width, fraction, strategy = case
    if strategy == 'joint' and args.smrt_dir is None:
        parser.error('Joint requires --smrt-dir')
    if strategy == 'transfer' and args.checkpoint is None:
        parser.error('Transfer requires --checkpoint')
    folds = args.folds if args.folds is not None else published_folds(case)
    if len(set(folds)) != len(folds):
        parser.error('Duplicate fold indices')
    heads, layers, ffn = ARCHITECTURES[width]
    for fold in folds:
        split_dir = args.target_dir / ('bm_scaffold' if scenario == 'bm' else scenario)
        command = [
            sys.executable, str(Path(__file__).with_name('target_only.py')),
            '--chromagrt-root', str(args.chromagrt_root), '--scenario', scenario,
            '--fold-dir', str(split_dir / f'fold_{fold}'), '--fold', str(fold),
            '--molecular-descriptors', str(args.target_dir / 'molecular_descriptors.tsv'),
            '--output', str(args.output_root / args.case / f'fold_{fold}'),
            '--strategy', 'target-only' if strategy == 'matched-updates' else strategy,
            '--fraction', fraction, '--accelerator', args.accelerator,
            '--hidden-dim', str(width), '--num-heads', str(heads),
            '--num-layers', str(layers), '--ffn-dim', str(ffn),
        ]
        if strategy == 'joint':
            command.extend(['--smrt-dir', str(args.smrt_dir)])
        if strategy == 'transfer':
            command.extend(['--checkpoint', str(args.checkpoint)])
        if strategy == 'matched-updates':
            command.append('--matched-updates')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
