from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_data_dir: str = "data/training_set"
    test_data_dir: str = "data/test_set"
    model_save_path: str = "dog_cat_classifier_model.keras"
