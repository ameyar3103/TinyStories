from dataclasses import dataclass, field
from typing import Dict
import sys

embed = 256
num_layer = 8
num_head = 4

@dataclass
class DataConfig:
    name: str = "tinystories"
    num_workers: int = 18
    batch_size: int = 8
    max_length: int = 512

@dataclass
class TokenizerConfig:
    name: str = "gpt_neo"
    cache_dir: str = (
        ".cache/tokenizer"
    )

@dataclass
class ModelConfig:
    name: str = "gpt2"
    gpt2: Dict[str, int] = field(
        default_factory=lambda: {"hidden_size": embed, "layers": num_layer, "heads": num_head}
    )
    compressor: Dict[str, int] = field(
        default_factory=lambda: {
            "n_positions": 512,
            "d_model": embed,
            "n_head": num_head,
            "n_layers": num_layer//2,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "compression_ratio": 2,  # 2 token will be conatenated going to d_model * 2 then scaled down by this ratio.
        }
    )
    bigger_transformer: Dict[str, int] = field(
        default_factory=lambda: {
            "d_model": embed,
            "n_head": num_head,
            "n_layers": num_layer,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        }
    )
    decompressor: Dict[str, int] = field(
        default_factory=lambda: {
            "n_positions": 512,
            "d_model": embed,
            "n_head": num_head,
            "n_layers": num_layer//2,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        }
    )


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01


@dataclass
class Config:
    cache_dir: str = ".cache"
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


config = Config()
