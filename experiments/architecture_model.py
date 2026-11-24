"""Configurable constructor; published forward/loss/optimizer stay inherited.

The constructor mirrors the pinned ChromaGRT constructor, in the same module
creation order, replacing only its four fixed sizes. No global monkeypatches
or edits to the published package are involved. Default initialization and
predictions are tested for exact parity with that package.
"""
import torch
from torch import nn
from lightning import pytorch as pl

from src.training.model_backends.chromagrt_model import ChromaGRTRegressor, ChromaGRTEncoderLayer
from src.training.model_backends.chromagrt_defaults import MAX_DISTANCE
from src.training.model_backends.chromagrt_dataset import TANAKA_COLUMNS
from model_architecture import Architecture
from program_duration_backend import ProgramDurationRegressor


class ConfigurableChromaGRTRegressor(ChromaGRTRegressor):
    def __init__(self, condition_dim, target_mean, target_std, config):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=['config'])
        architecture = Architecture.from_config(config)
        self.config = dict(config, **architecture.as_dict())
        hidden_dim, num_heads = architecture.hidden_dim, architecture.num_heads
        ffn_dim = architecture.ffn_dim
        dropout = self.config['dropout']
        self.target_mean, self.target_std = float(target_mean), float(target_std)
        self.tanaka_block_dropout = float(self.config.get('tanaka_block_dropout', 0.0))
        if not 0 <= self.tanaka_block_dropout <= 1:
            raise ValueError('tanaka_block_dropout must be between 0 and 1')
        self.tanaka_value_indices = tuple(int(i) for i in self.config.get('tanaka_value_indices', ()))
        self.tanaka_missing_indices = tuple(int(i) for i in self.config.get('tanaka_missing_indices', ()))
        if self.tanaka_block_dropout > 0:
            if len(self.tanaka_value_indices) != len(TANAKA_COLUMNS) or len(self.tanaka_missing_indices) != len(TANAKA_COLUMNS):
                raise ValueError('Expected six Tanaka values and six missingness indices')
            indices = (*self.tanaka_value_indices, *self.tanaka_missing_indices)
            if len(set(indices)) != len(indices) or any(i < 0 or i >= condition_dim for i in indices):
                raise ValueError('Tanaka indices must be distinct and inside the condition input')

        self.atom_encoder = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(6, hidden_dim, padding_idx=0)
        self.formal_charge_encoder = nn.Embedding(7, hidden_dim)
        self.hybridization_encoder = nn.Embedding(6, hidden_dim, padding_idx=0)
        self.aromatic_encoder = nn.Embedding(2, hidden_dim)
        self.total_hs_encoder = nn.Embedding(5, hidden_dim, padding_idx=0)
        self.distance_bias = nn.Embedding(MAX_DISTANCE + 1, num_heads)
        self.condition_node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.condition_node_bias = nn.Parameter(torch.zeros(num_heads))
        self.layers = nn.ModuleList([
            ChromaGRTEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(architecture.num_layers)])
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, 1))


class ConfigurableProgramDurationRegressor(ConfigurableChromaGRTRegressor, ProgramDurationRegressor):
    """Configurable constructor with inherited duration target/loss/decoding."""
