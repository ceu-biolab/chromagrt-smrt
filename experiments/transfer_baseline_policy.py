"""The published scratch optimization policy, used after SMRT initialization."""
from src.training.model_backends.chromagrt_defaults import (
    DEFAULT_CHROMAGRT_CONFIG, EARLY_STOPPING_PATIENCE, WEIGHT_DECAY,
)


POLICY = dict(
    name='baseline', optimizer='AdamW',
    lr=DEFAULT_CHROMAGRT_CONFIG['lr'], weight_decay=WEIGHT_DECAY,
    warmup_epochs=0, molecular_freeze_epochs=0,
    plateau_factor=DEFAULT_CHROMAGRT_CONFIG['plateau_factor'],
    plateau_patience=DEFAULT_CHROMAGRT_CONFIG['plateau_patience'],
    plateau_min_lr=DEFAULT_CHROMAGRT_CONFIG['plateau_min_lr'],
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    monitor='val_mae', monitor_units='seconds',
    optimizer_state='fresh in every fold',
)


def validate_config(config):
    """Reject a policy/configuration mismatch before starting training."""
    if config.get('transfer_lr_profile') != POLICY['name']:
        raise ValueError('This backend requires --transfer-lr-profile baseline')
    if config.get('transfer_lr_policy') != POLICY:
        raise ValueError('The recorded transfer policy differs from baseline')
    for key in ('lr', 'plateau_factor', 'plateau_patience', 'plateau_min_lr'):
        if config.get(key) != POLICY[key]:
            raise ValueError(f'Baseline transfer requires {key}={POLICY[key]}')
