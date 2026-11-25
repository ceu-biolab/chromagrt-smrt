"""SMRT checkpoint validation and parameter initialization."""
import hashlib
import json
from pathlib import Path

import torch

from model_architecture import Architecture
from program_duration import FEATURE


MOLECULAR_PREFIXES = ('atom_encoder.', 'degree_encoder.', 'formal_charge_encoder.',
    'hybridization_encoder.', 'aromatic_encoder.', 'total_hs_encoder.', 'distance_bias.', 'layers.')


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def molecular_parameter(name):
    return name.startswith(MOLECULAR_PREFIXES)


def input_encoding(raw_train, config, audit):
    """Describe the actual published transforms; never fit on held-out rows."""
    result = {}
    molecular = dict(zip(('mono_iso_mass','xlogp'), zip(audit['molecular_descriptor_means'],
                                                     audit['molecular_descriptor_scales'])))
    standard = config['condition_normalization'] == 'standard'
    condition = {}
    if standard:
        columns = raw_train.loc[:, 'column.length':].columns
        statistics = audit['condition_statistics']
        if len(columns) != len(statistics['mean']):
            raise ValueError('Condition scaler columns do not match preprocessing audit')
        condition = dict(zip(columns, zip(statistics['mean'], statistics['scale'])))
    for column in audit['condition_columns']:
        scale, offset, kind = 1., 0., 'affine'
        if column in molecular:
            mean, std = molecular[column]
            scale, offset = 1./std, -mean/std
        elif column == FEATURE:
            duration = audit['rt_normalization']
            std = duration['duration_feature_scale_minutes']
            scale, offset = 1./(60*std), -duration['duration_feature_mean_minutes']/std
        elif column in condition:
            mean, std = condition[column]
            scale, offset = 1./std, -mean/std
        elif not standard:
            if column.startswith('eluent.') and not column.endswith('.pH'):
                kind = 'presence'
            elif column == 'column.length': scale = 1./250
            elif column == 'column.temperature' or column.startswith('B [%]_'): scale = 1./100
            elif column.startswith('t [min]_'): scale = 1./60
            elif column.startswith('eluent.') and column.endswith('.pH'): scale = 1./14
        result[column] = dict(kind=kind, scale=float(scale), offset=float(offset))
    return result


def checkpoint_metadata(path, allow_smoke=False):
    path = Path(path).resolve()
    manifest = path.with_name('pretrain_checkpoint.json')
    value = json.loads(manifest.read_text())
    if value.get('format') != 'chromagrt-smrt-v1' or value.get('source_dataset') != 'SMRT_0186':
        raise ValueError('Not a supported SMRT checkpoint')
    if not value.get('complete') or value.get('weights_sha256') != digest(path):
        raise ValueError('Incomplete or modified SMRT checkpoint')
    if value.get('smoke') and not allow_smoke:
        raise ValueError('A smoke checkpoint cannot initialize a full experiment')
    if value.get('scaffold_filtering') != 'none; one common checkpoint for all folds':
        raise ValueError('Unexpected SMRT scaffold protocol')
    return value


def initialize_transfer(model, path, columns, encoding, mode, allow_smoke=False):
    metadata = checkpoint_metadata(path, allow_smoke)
    if metadata['architecture'] != Architecture.from_config(model.config).as_dict():
        raise ValueError('SMRT checkpoint architecture differs from the requested model')
    if metadata['rt_normalization'] != model.config['rt_normalization']:
        raise ValueError('Checkpoint and target RT normalization must match')
    if list(columns) != metadata['condition_columns']:
        raise ValueError('Checkpoint condition columns/order differ from target schema')
    if mode not in ('molecular','full'):
        raise ValueError('Transfer mode must be molecular or full')
    source = torch.load(path, map_location='cpu', weights_only=True)
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError('Checkpoint/model parameter names differ')
    if any(source[k].shape != target[k].shape for k in source):
        raise ValueError('Checkpoint/model parameter shapes differ')
    selected = [name for name in source if mode == 'full' or molecular_parameter(name)]
    for name in selected:
        target[name] = source[name].clone()
    report = dict(mode=mode, weights_sha256=metadata['weights_sha256'], loaded=selected,
                  reinitialized=sorted(set(target)-set(selected)), affine_columns=[],
                  nonaffine_columns_requires_finetuning=[], missing_metadata_policy=metadata['missing_metadata_policy'])
    if mode == 'full':
        factors, offsets = [], []
        for column in columns:
            old, new = metadata['input_encoding'][column], encoding[column]
            if old['kind'] == new['kind'] == 'affine':
                factor = old['scale']/new['scale']
                offset = old['offset']-factor*new['offset']
                report['affine_columns'].append(column)
            elif old['kind'] == new['kind'] == 'presence':
                factor, offset = 1., 0.
            else:
                # Presence loses concentration information: there is no exact
                # affine inverse. Keep that weight, explicitly audit the change,
                # and adapt the condition block at its higher fine-tuning LR.
                factor, offset = 1., 0.
                report['nonaffine_columns_requires_finetuning'].append(column)
            factors.append(factor); offsets.append(offset)
        weight = target['condition_encoder.0.weight'].clone()
        target['condition_encoder.0.weight'] = weight * weight.new_tensor(factors)
        target['condition_encoder.0.bias'] += weight @ weight.new_tensor(offsets)
        if model.config['rt_normalization'] == 'none':
            ratio = metadata['target_std']/model.target_std
            target['regression_head.3.weight'] *= ratio
            target['regression_head.3.bias'] = (target['regression_head.3.bias']*ratio
                + (metadata['target_mean']-model.target_mean)/model.target_std)
    if not all(torch.isfinite(value).all() for value in target.values()):
        raise ValueError('Nonfinite transferred weights')
    model.load_state_dict(target, strict=True)
    return report
