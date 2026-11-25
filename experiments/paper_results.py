"""Regenerate the paper's metrics and availability figure from saved predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_runs import cases, name, published_folds

KEYS = ['molecule_id', 'cc_id', 'rt']


def errors(frame):
    return np.abs(frame.pred_rt.to_numpy() - frame.rt.to_numpy())


def metrics(values):
    return np.mean(values), np.sqrt(np.mean(np.square(values)))


def aggregate(frames):
    fold_metrics = np.array([metrics(errors(frame)) for frame in frames.values()])
    mae, rmse = metrics(np.concatenate([errors(frame) for frame in frames.values()]))
    return dict(mae=mae, rmse=rmse, mae_sd=fold_metrics[:, 0].std(ddof=1),
                rmse_sd=fold_metrics[:, 1].std(ddof=1))


def read_predictions(root, case):
    paths = root / name(case)
    result = {}
    for fold in published_folds(case):
        frame = pd.read_csv(paths / f'fold_{fold}' / 'Results.tsv', sep='\t')
        if frame.molecule_id.duplicated().any() or not np.isfinite(errors(frame)).all():
            raise ValueError(f'Invalid predictions for {name(case)}, fold {fold}')
        result[fold] = frame.sort_values(KEYS).reset_index(drop=True)
    return result


def availability_figure(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from matplotlib.transforms import ScaledTranslation

    fractions = ('0.25', '0.5', '1')
    x = 0.8 * 46557 * np.array([float(fraction) for fraction in fractions])
    def values(strategy, metric):
        selected = fractions[:1] if strategy == 'matched-updates' else fractions
        return np.array([summary[('random', 256, f, strategy)][metric] for f in selected])

    slope, intercept = np.polyfit(np.log(x), np.log(values('target-only', 'mae')), 1)
    joint_mae = values('joint', 'mae')
    equal_size = np.exp((np.log(joint_mae) - intercept) / slope)
    pd.DataFrame(dict(fraction=fractions, nominal_rows=x, equivalent_rows=equal_size,
                      additional_rows=equal_size - x, extrapolated=equal_size > x[-1])
                 ).to_csv(output / 'effective-data.tsv', sep='\t', index=False)
    with plt.rc_context({'font.size': 7, 'pdf.fonttype': 42, 'axes.linewidth': 0.6}):
        fig, ax = plt.subplots(figsize=(4.5, 2.3))
        fig.subplots_adjust(left=.115, right=.978, top=.77, bottom=.22)
        ax.set(xscale='log', yscale='log', xlim=(8000, 45000), ylim=(20.4, 41.8))
        for start, end, style in [(x[0], x[-1], '--'), (x[-1], 42000, ':')]:
            points = np.geomspace(start, end, 120)
            ax.plot(points, np.exp(intercept + slope * np.log(points)), style, color='#555555', lw=1)
        for observed_size, estimated_size, mae in zip(x, equal_size, joint_mae):
            ax.annotate('', xy=(estimated_size, mae), xytext=(observed_size, mae),
                        arrowprops=dict(arrowstyle='-|>', color='#434343', lw=.9,
                                        mutation_scale=7, shrinkA=2, shrinkB=0))
        for strategy, label, color, marker, offset in [
                ('target-only', 'Target-only', '#252525', 'o', -3),
                ('transfer', 'Pretraining', '#0072B2', 's', 3),
                ('joint', 'Joint', '#CB5200', '^', 0),
                ('matched-updates', 'Target-only, matched update cap', '#006F55', 'D', 0)]:
            mean = values(strategy, 'mae')
            ax.errorbar(x[:len(mean)], mean, yerr=values(strategy, 'mae_sd'),
                        color=color, fmt=marker, mfc='white' if marker == 'D' else color,
                        ms=3.8, mew=.85, elinewidth=.75, capsize=1.7, label=label,
                        transform=ax.transData + ScaledTranslation(offset / 72, 0, fig.dpi_scale_trans))
        ax.set_xticks(x, ['9,311 (25%)', '18,623 (50%)', '37,246 (100%)'])
        ax.set_yticks([22, 25, 30, 35, 40], ['22', '25', '30', '35', '40'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('RepoRT training observations per fold (approx.)')
        ax.set_ylabel('MAE (s)')
        ax.grid(axis='y', color='#dfdfdf', lw=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.035), ncol=2, frameon=False,
                  columnspacing=1.8, borderaxespad=0)
        fig.savefig(output / 'effective-data.pdf')
        fig.savefig(output / 'effective-data.png', dpi=600)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--random-target', type=Path)
    parser.add_argument('--smrt-train', type=Path)
    args = parser.parse_args()
    if (args.random_target is None) != (args.smrt_train is None):
        parser.error('Provide both identity inputs to regenerate the overlap analysis')
    predictions = {case: read_predictions(args.predictions, case) for case in cases()}
    summary, paired = {}, []
    for case, folds in predictions.items():
        reference = predictions[(*case[:3], 'target-only')]
        summary[case] = aggregate(folds)
        baseline = aggregate({fold: reference[fold] for fold in folds})
        summary[case].update(folds=len(folds), delta_mae_pct=100 * (
            summary[case]['mae'] / baseline['mae'] - 1))
        for fold, frame in folds.items():
            pd.testing.assert_frame_equal(frame[KEYS], reference[fold][KEYS])
            mae, rmse = metrics(errors(frame))
            ref_mae, ref_rmse = metrics(errors(reference[fold]))
            paired.append(dict(case=name(case), fold=fold, n=len(frame), mae=mae, rmse=rmse,
                               delta_mae=mae-ref_mae, delta_rmse=rmse-ref_rmse))
    joint_case = ('random', 256, '0.25', 'joint')
    control_case = (*joint_case[:3], 'matched-updates')
    joint = summary[joint_case]
    control = summary[control_case]
    wins = sum(errors(predictions[joint_case][fold]).mean()
               < errors(predictions[control_case][fold]).mean()
               for fold in published_folds(joint_case))
    control_comparison = dict(
        joint_mae=joint['mae'], control_mae=control['mae'],
        joint_rmse=joint['rmse'], control_rmse=control['rmse'],
        joint_delta_mae_pct=100 * (joint['mae'] / control['mae'] - 1),
        folds_favoring_joint=wins,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([control_comparison]).to_csv(
        args.output / 'matched-budget-comparison.tsv', sep='\t', index=False)
    pd.DataFrame([dict(case=name(case), **row) for case, row in summary.items()]
                 ).to_csv(args.output / 'summary.tsv', sep='\t', index=False)
    pd.DataFrame(paired).to_csv(args.output / 'paired-folds.tsv', sep='\t', index=False)
    availability_figure(summary, args.output)
    if args.random_target:
        source = pd.read_csv(args.smrt_train, sep='\t', usecols=['inchi.std'])['inchi.std']
        rows = []
        for fraction in ('0.25', '0.5', '1'):
            base = ('random', 256, fraction)
            for fold in range(10):
                test = pd.read_csv(args.random_target / 'random' / f'fold_{fold}' / 'test_data.tsv',
                                   sep='\t', usecols=['molecule_id', 'inchi.std']).set_index('molecule_id')
                for strategy in ('target-only', 'joint'):
                    frame = predictions[(*base, strategy)][fold]
                    identities = test['inchi.std'].reindex(frame.molecule_id)
                    if identities.isna().any():
                        raise ValueError('Missing target test identities')
                    present = identities.isin(source).to_numpy()
                    for mask, group in [(present, 'present'), (~present, 'absent')]:
                        rows.append(dict(fraction=fraction, strategy=strategy, group=group,
                                         n=mask.sum(), absolute_error=errors(frame)[mask].sum()))
        totals = pd.DataFrame(rows).groupby(['fraction', 'strategy', 'group'], as_index=False).sum()
        totals['mae'] = totals.absolute_error / totals.n
        totals.to_csv(args.output / 'smrt-identity-overlap.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
