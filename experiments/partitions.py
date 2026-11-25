"""Select the nested RepoRT subsets and the auxiliary training records."""

from decimal import Decimal, ROUND_FLOOR
import hashlib

import pandas as pd
from torch.utils.data import Sampler
import torch


def rank_rows(frame, fold, seed):
    if frame.molecule_id.duplicated().any() or frame.cc_id.isna().any():
        raise ValueError('Unique observation IDs and nonmissing conditions required')
    rows = []
    for cc, group in frame.groupby('cc_id', sort=True):
        ordered = sorted(group.molecule_id.tolist(), key=lambda ident:
            (hashlib.sha256(f'{seed}|{fold}|{cc}|{ident}'.encode()).hexdigest(), ident))
        rows.extend((ident, cc, rank, len(ordered)) for rank, ident in enumerate(ordered))
    return pd.DataFrame(rows, columns=['molecule_id', 'cc_id', 'rank', 'method_train_n'])


def target_subset(frame, fold, fraction):
    ranks = rank_rows(frame, fold, seed=42)
    counts = {
        int(n): max(1, int((Decimal(fraction) * int(n)).to_integral_value(rounding=ROUND_FLOOR)))
        for n in ranks.method_train_n.unique()
    }
    ids = ranks.loc[ranks['rank'] < ranks.method_train_n.map(counts), 'molecule_id']
    return frame.loc[frame.molecule_id.isin(ids)].copy()


def scaffold_column(rows):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    cache = {}
    for smiles in rows['smiles.std'].unique():
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if molecule is None:
            raise ValueError(f'Invalid curated SMILES: {smiles}')
        cache[smiles] = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return rows['smiles.std'].map(cache)


def filter_smrt(smrt, validation, test):
    from rdkit import rdBase

    if rdBase.rdkitVersion != '2025.09.6':
        raise RuntimeError('The paper BM filter requires RDKit 2025.09.6')
    held_out = pd.concat([validation, test], ignore_index=True)
    scaffolds = scaffold_column(smrt)
    held_scaffolds = set(scaffold_column(held_out)) - {''}
    identity_match = smrt['inchi.std'].isin(held_out['inchi.std'])
    if (scaffolds.eq('') & identity_match).any():
        raise ValueError('An empty-scaffold SMRT molecule overlaps a held-out identity')
    return smrt.loc[~(scaffolds.isin(held_scaffolds) | identity_match)].copy()


def append_smrt(parts, smrt, scenario):
    if set(smrt.columns) != set(parts['train'].columns):
        raise ValueError('The SMRT and RepoRT feature schemas differ')
    if len(smrt) != 70184 or not smrt.molecule_id.str.startswith('0186_').all():
        raise ValueError('Expected the frozen 70,184-row SMRT training partition')
    if scenario == 'bm':
        smrt = filter_smrt(smrt, parts['valid'], parts['test'])
    training = pd.concat([parts['train'], smrt[parts['train'].columns].assign(split='train')],
                         ignore_index=True)
    if training.molecule_id.duplicated().any():
        raise ValueError('Repeated observation IDs in combined training data')
    return training


class RepeatedShuffle(Sampler):
    """Draw a Joint-sized epoch from shuffled permutations of target rows."""

    def __init__(self, population, draws):
        if not 0 < population <= draws:
            raise ValueError('Invalid target population or epoch size')
        self.population = population
        self.draws = draws

    def __len__(self):
        return self.draws

    def __iter__(self):
        for offset in range(0, self.draws, self.population):
            order = torch.randperm(self.population)
            yield from order[:self.draws - offset].tolist()
