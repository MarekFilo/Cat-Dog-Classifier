from dataclasses import dataclass
import tensorflow as tf

tf.random.set_seed(123)


@dataclass
class ModelConfig:
    input_shape: tuple = (256, 256, 3)
    batch_size: int = 32
    epochs: int = 10
    patience: int = 4
    learning_rate: float = 0.001
    dropout_rate: float = 0
