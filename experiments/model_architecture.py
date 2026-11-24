"""Validated, serializable architecture settings for the experimental runner."""
from dataclasses import asdict, dataclass

from src.training.model_backends.chromagrt_defaults import HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM


@dataclass(frozen=True)
class Architecture:
    hidden_dim: int = HIDDEN_DIM
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    ffn_dim: int = FFN_DIM

    def __post_init__(self):
        for name, value in self.as_dict().items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden_dim % self.num_heads:
            raise ValueError('--hidden-dim must be divisible by --num-heads')

    def as_dict(self):
        return asdict(self)

    @property
    def label(self):
        return f'h{self.hidden_dim}_heads{self.num_heads}_layers{self.num_layers}_ffn{self.ffn_dim}'

    @classmethod
    def from_config(cls, config):
        defaults = cls().as_dict()
        values = {key: config.get(key, value) for key, value in defaults.items()}
        if 'ffn_dim' not in config:
            values['ffn_dim'] = 2 * values['hidden_dim']
        return cls(**values)

    @classmethod
    def from_args(cls, args):
        # Preserve the published 2:1 FFN/embedding ratio unless explicitly set.
        return cls(args.hidden_dim, args.num_heads, args.num_layers,
                   args.ffn_dim if args.ffn_dim is not None else 2 * args.hidden_dim)

    def cli_args(self):
        return [argument for key, value in self.as_dict().items()
                for argument in ('--' + key.replace('_', '-'), str(value))]


def add_arguments(parser):
    parser.add_argument('--hidden-dim', type=int, default=HIDDEN_DIM,
                        help='Atom and condition embedding width (default: 256)')
    parser.add_argument('--num-heads', type=int, default=NUM_HEADS,
                        help='Attention heads; must divide hidden-dim (default: 8)')
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS,
                        help='Transformer layers (default: 6)')
    parser.add_argument('--ffn-dim', type=int,
                        help='FFN and regression hidden width (default: 2 * hidden-dim)')
