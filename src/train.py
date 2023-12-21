import os
import tensorflow as tf
from model.model import create_model
from config.train_config import TrainConfig
from config.model_config import ModelConfig
from tensorflow.keras.callbacks import EarlyStopping


def load_data(model_config: ModelConfig, train_config: TrainConfig):
    train_data = tf.keras.utils.image_dataset_from_directory(
        train_config.train_data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(model_config.input_shape[0], model_config.input_shape[1]),
        batch_size=model_config.batch_size,
        validation_split=0.2,
        subset="training",
        seed=123,
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        train_config.test_data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(model_config.input_shape[0], model_config.input_shape[1]),
        batch_size=model_config.batch_size,
        validation_split=0.2,
        subset="validation",
        seed=123,
    )

    return train_data, test_data


def train_model(model_config: ModelConfig, train_config: TrainConfig):
    train_data, test_data = load_data(model_config, train_config)

    model = create_model(model_config)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=model_config.patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_data,
        epochs=model_config.epochs,
        validation_data=test_data,
        verbose=True,
        callbacks=[early_stopping],
    )

    model.save(train_config.model_save_path)


if __name__ == "__main__":
    model_config = ModelConfig()
    train_config = TrainConfig()

    train_model(model_config, train_config)
