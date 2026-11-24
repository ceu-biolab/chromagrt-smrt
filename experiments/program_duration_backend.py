"""Opt-in duration-normalized ChromaGRT, outside the immutable published src.

The encoder, regression head, Tanaka dropout and optimizer are inherited from
the published model. Only target encoding, seconds loss and decoding differ.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.training.model_backends import chromagrt_backend as published
from src.training.model_backends.chromagrt_dataset import ChromaGRTDataset, TargetScaler, collate_chromagrt
from src.training.model_backends.chromagrt_model import ChromaGRTRegressor
from program_duration import DURATION, MODE


class ProgramDurationDataset(ChromaGRTDataset):
    def __init__(self, df, condition_columns, max_distance, predicting=False):
        frame = df.copy()
        duration = frame[DURATION].to_numpy(dtype=float)
        if not np.isfinite(duration).all() or (duration <= 0).any():
            raise ValueError('Invalid unscaled program duration')
        # Prediction never reads experimental RTs, even if they are supplied.
        frame['rt'] = 0.0 if predicting else frame['rt'] / duration
        super().__init__(frame, condition_columns, TargetScaler(mean=0.0, std=1.0), max_distance)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item[DURATION] = torch.tensor([self.df.iloc[index][DURATION]], dtype=torch.float32)
        return item


def collate_duration(batch):
    result = collate_chromagrt(batch)
    result[DURATION] = torch.stack([item[DURATION] for item in batch])
    return result


class ProgramDurationRegressor(ChromaGRTRegressor):
    def training_step(self, batch, batch_idx):
        # Same absolute-error objective for every method, in seconds: no 1/D
        # weighting of the loss and no clipping of normalized RTs to [0, 1].
        error_seconds = (self(batch) - batch['target']) * batch[DURATION]
        loss = F.l1_loss(error_seconds, torch.zeros_like(error_seconds))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        difference = self(batch) - batch['target']
        loss = (difference.abs() * batch[DURATION]).mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae_normalized', difference.abs().mean(), on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch) * batch[DURATION]


def build_loader(frame, condition_columns, config, shuffle=False, predicting=False):
    return DataLoader(ProgramDurationDataset(frame, condition_columns, published.MAX_DISTANCE, predicting),
        batch_size=config['batch_size'], shuffle=shuffle, num_workers=config['num_workers'], collate_fn=collate_duration)


def train_chromagrt(train_df, val_df, param_dict, results_path, using_moldescs=False, save_model=True):
    config = published._config_from_param_dict(param_dict)
    if config.get('rt_normalization') != MODE:
        raise ValueError('Duration backend requires --rt-normalization program-duration')
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
    model_args = dict(condition_dim=len(columns), target_mean=0.0, target_std=1.0, config=config)
    model = ProgramDurationRegressor(**model_args)
    checkpoint = ModelCheckpoint(dirpath=results_path, filename='best-duration-{epoch}-{val_mae:.4f}',
                                 monitor='val_mae', mode='min', save_top_k=1)
    trainer = pl.Trainer(logger=CSVLogger(save_dir=results_path, name='lightning_logs'),
        enable_progress_bar=True, log_every_n_steps=1, accelerator=config['accelerator'], devices=1,
        max_epochs=config['max_epochs'], callbacks=[
            EarlyStopping(monitor='val_mae', mode='min', patience=published.EARLY_STOPPING_PATIENCE),
            checkpoint, LearningRateMonitor(logging_interval='epoch')])
    trainer.fit(model, build_loader(train_df, columns, config, shuffle=True), build_loader(val_df, columns, config))
    if checkpoint.best_model_path:
        model = ProgramDurationRegressor.load_from_checkpoint(checkpoint.best_model_path, **model_args)
    if save_model:
        torch.save(model.state_dict(), os.path.join(results_path, 'model.pt'))
    for name in os.listdir(results_path):
        if name.endswith('.ckpt'):
            os.unlink(os.path.join(results_path, name))
    return dict(backend_name='chromagrt', model=model, trainer=trainer, config=config,
                condition_columns=columns, rt_normalization=MODE)


def predict_chromagrt(bundle, frame):
    loader = build_loader(frame, bundle['condition_columns'], bundle['config'], predicting=True)
    predictions = bundle['trainer'].predict(bundle['model'], loader)
    return np.concatenate([batch.cpu().numpy() for batch in predictions], axis=0)
