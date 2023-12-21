import tensorflow as tf
from model.model import create_model
from config.train_config import TrainConfig
from config.model_config import ModelConfig


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_image(model, image_path, model_config):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=model_config.input_shape[:2]
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normalize to [0,1]

    predictions = model.predict(img_array)

    return predictions[0][0] > 0.5


if __name__ == "__main__":
    model_config = ModelConfig()
    train_config = TrainConfig()

    trained_model = load_model(train_config.model_save_path)

    image_path = "test_image.jpg"

    is_Cat = predict_image(trained_model, image_path, model_config)

    if is_Cat:
        print("Cat")
    else:
        print("Dog")
