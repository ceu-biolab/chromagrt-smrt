"""Training/prediction adapter for either target mode and configurable sizes."""
import os

import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.training.model_backends import chromagrt_backend as published
from src.training.model_backends.chromagrt_dataset import build_target_scaler
from model_architecture import Architecture
from architecture_model import ConfigurableChromaGRTRegressor, ConfigurableProgramDurationRegressor
from program_duration import MODE
from program_duration_backend import build_loader as duration_loader


def build_loader(frame, columns, scaler, config, shuffle=False, predicting=False):
    if config['rt_normalization'] == MODE:
        return duration_loader(frame, columns, config, shuffle=shuffle, predicting=predicting)
    if predicting:
        frame = frame.copy()
        frame['rt'] = 0.0
    return published._build_loader(frame, columns, scaler, config, shuffle=shuffle)


def train_chromagrt(train_df, val_df, param_dict, results_path, using_moldescs=False, save_model=True,
                    training_sampler=None, pretrained_checkpoint=None):
    config = published._config_from_param_dict(param_dict)
    architecture = Architecture.from_config(config)
    config.update(architecture.as_dict())
    mode = config.get('rt_normalization', 'none')
    if mode not in ('none', MODE):
        raise ValueError(f'Unknown RT normalization: {mode}')
    config['rt_normalization'] = mode
    if config.get('seed') is not None:
        pl.seed_everything(int(config['seed']), workers=True)
    columns = published.get_condition_columns(train_df, using_moldescs=using_moldescs,
        include_tanaka=config.get('use_tanaka', True),
        excluded_blocks=config.get('excluded_condition_blocks', ()),
        excluded_molecular_descriptors=config.get('excluded_molecular_descriptors', ()))
    dropout = float(config.get('tanaka_block_dropout', 0.0))
    if not 0 <= dropout <= 1:
        raise ValueError('tanaka_block_dropout must be between 0 and 1')
    if dropout:
        if not config.get('use_tanaka', True) or config['condition_normalization'] != 'deterministic':
            raise ValueError('Tanaka block dropout requires Tanaka and deterministic normalization')
        values, missing = published.resolve_tanaka_indices(columns)
        metadata = dict(tanaka_value_indices=values, tanaka_missing_indices=missing,
            metadata_tanaka_block_dropout_scope='per_observation', metadata_tanaka_block_dropout_training_only=True)
        config.update(metadata)
        param_dict.update(metadata)
    scaler = None if mode == MODE else build_target_scaler(train_df)
    model_class = ConfigurableProgramDurationRegressor if mode == MODE else ConfigurableChromaGRTRegressor
    model_args = dict(condition_dim=len(columns), target_mean=0.0 if scaler is None else scaler.mean,
                      target_std=1.0 if scaler is None else scaler.std, config=config)
    # Match the published initialization order: loaders, then model.
    train_loader = build_loader(train_df, columns, scaler, config, shuffle=True)
    valid_loader = build_loader(val_df, columns, scaler, config)
    if training_sampler is not None:
        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_loader.dataset, batch_size=train_loader.batch_size,
            collate_fn=train_loader.collate_fn, sampler=training_sampler,
        )
    model = model_class(**model_args)
    if pretrained_checkpoint is not None:
        import json
        from pathlib import Path

        from transfer_baseline_policy import POLICY, validate_config
        from transfer_core import initialize_transfer

        config.update(transfer_lr_profile=POLICY['name'], transfer_lr_policy=POLICY)
        validate_config(config)
        audit = initialize_transfer(model, pretrained_checkpoint, columns, {}, 'molecular')
        audit['optimization'] = POLICY
        (Path(results_path) / 'transfer_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    info = dict(architecture=architecture.as_dict(), architecture_label=architecture.label,
                metadata_n_parameters=sum(p.numel() for p in model.parameters()))
    config.update(info)
    param_dict.update(info)
    # Persist the sizes in Lightning's checkpoint as well as model.pt's sidecar.
    model.config.update(info)
    model.save_hyperparameters({'config': config})
    checkpoint = ModelCheckpoint(dirpath=results_path, filename='best-architecture-{epoch}-{val_mae:.4f}',
                                 monitor='val_mae', mode='min', save_top_k=1)
    trainer = pl.Trainer(logger=CSVLogger(save_dir=results_path, name='lightning_logs'),
        enable_progress_bar=True, log_every_n_steps=1, accelerator=config['accelerator'], devices=1,
        max_epochs=config['max_epochs'], callbacks=[
            EarlyStopping(monitor='val_mae', mode='min', patience=published.EARLY_STOPPING_PATIENCE),
            checkpoint, LearningRateMonitor(logging_interval='epoch')])
    trainer.fit(model, train_loader, valid_loader)
    if checkpoint.best_model_path:
        model = model_class.load_from_checkpoint(checkpoint.best_model_path, **model_args)
    if save_model:
        torch.save(model.state_dict(), os.path.join(results_path, 'model.pt'))
    for name in os.listdir(results_path):
        if name.endswith('.ckpt'):
            os.unlink(os.path.join(results_path, name))
    return dict(backend_name='chromagrt', model=model, trainer=trainer, config=config,
                condition_columns=columns, target_scaler=scaler)


def predict_chromagrt(bundle, frame):
    loader = build_loader(frame, bundle['condition_columns'], bundle['target_scaler'], bundle['config'], predicting=True)
    batches = bundle['trainer'].predict(bundle['model'], loader)
    return np.concatenate([batch.cpu().numpy() for batch in batches], axis=0)
