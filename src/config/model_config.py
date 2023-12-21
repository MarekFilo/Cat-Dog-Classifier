from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_shape: tuple = (128, 128, 3)
    batch_size: int = 32
    epochs: int = 10
