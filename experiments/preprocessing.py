import numpy as np

def scaled_inputs(parts, config, descriptor_path):
    """Use the unchanged article helpers, fitting exclusively on training."""
    from src.training.functions.splitted_sets_functions import (
        add_moldescs, get_scaled_moldescs_train, get_scaled_moldesc_testval,
        get_scaled_input_train_data, get_scaled_datasets, get_deterministic_input_data,
        _fit_fold_metadata_imputer,
    )
    raw = {role: add_moldescs(frame, str(descriptor_path)) for role, frame in parts.items()}
    for role in parts:
        if list(raw[role].molecule_id) != list(parts[role].molecule_id):
            raise AssertionError('Descriptor join changed observation identities or order')
    scaled_train, molecular_scaler = get_scaled_moldescs_train(raw['train'])
    scaled = {'train': scaled_train}
    scaled.update({role: get_scaled_moldesc_testval(raw[role], molecular_scaler) for role in ('valid', 'test')})
    means = _fit_fold_metadata_imputer(scaled['train'])
    if config['condition_normalization'] == 'standard':
        scaled['train'], condition_scaler = get_scaled_input_train_data(scaled['train'])
        for role in ('valid', 'test'):
            scaled[role] = get_scaled_datasets(scaled[role], condition_scaler)
        condition_stats = dict(mean=condition_scaler.mean_.tolist(), scale=condition_scaler.scale_.tolist())
    else:
        # The "split" label is required to fit/reset the article deterministic imputer.
        for role in ('train', 'valid', 'test'):
            scaled[role] = get_deterministic_input_data(scaled[role])
        condition_stats = dict(normalization='deterministic; fixed divisors, train-only missing-value means')
    from src.training.model_backends.chromagrt_dataset import get_condition_columns
    condition_columns = get_condition_columns(scaled['train'], using_moldescs=True, include_tanaka=True)
    for role, frame in scaled.items():
        if not np.isfinite(frame[condition_columns].to_numpy(dtype=float)).all():
            raise AssertionError(f'Invalid model inputs after fold preprocessing: {role}')
    audit = dict(fit_partition='train', n_training_rows=len(parts['train']),
                 n_training_conditions=parts['train'].cc_id.nunique(), metadata_means=means,
                 molecular_descriptor_means=molecular_scaler.mean_.tolist(),
                 molecular_descriptor_scales=molecular_scaler.scale_.tolist(),
                 condition_columns=condition_columns, condition_statistics=condition_stats,
                 missing_values_before={role: int(frame[condition_columns].isna().sum().sum()) for role, frame in raw.items()},
                 all_model_inputs_finite=True)
    return raw, scaled, audit
