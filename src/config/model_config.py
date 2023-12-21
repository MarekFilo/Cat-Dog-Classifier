from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_shape: tuple = (256, 256, 3)
    batch_size: int = 32
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 0.001
    droput_rate: float = 0
