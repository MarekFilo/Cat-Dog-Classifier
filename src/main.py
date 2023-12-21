import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
from model.model import create_model
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from predict import load_model, predict_image
from train import train_model
import os


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat-Dog Classifier")
        self.root.geometry("300x200")

        self.model_config = ModelConfig()
        self.train_config = TrainConfig()

        self.load_model_button = tk.Button(
            root, text="Load Model", command=self.load_model
        )
        self.load_model_button.pack(pady=10)

        self.train_model_button = tk.Button(
            root, text="Train Model", command=self.train_model_threaded
        )
        self.train_model_button.pack(pady=10)

        self.predict_button = tk.Button(
            root, text="Predict Image", command=self.predict_image
        )
        self.predict_button.pack(pady=10)

        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def load_model(self):
        trained_model_path = self.train_config.model_save_path
        if os.path.exists(trained_model_path):
            self.trained_model = load_model(trained_model_path)
            self.status_label["text"] = "Model loaded successfully!"
        else:
            self.status_label[
                "text"
            ] = "Trained model not found. Train the model first."

    def train_model_threaded(self):
        thread = threading.Thread(target=self.train_model)
        thread.start()
        self.root.after(100, self.check_training_status, thread)

    def train_model(self):
        self.status_label["text"] = "Training model..."
        train_model(self.model_config, self.train_config)
        self.status_label["text"] = "Model trained successfully!"

    def check_training_status(self, thread):
        if thread.is_alive():
            dots = "." * (self.status_label["text"].count(".") % 3 + 1)
            self.status_label["text"] = f"Training model{dots}"
            self.root.after(100, self.check_training_status, thread)

    def predict_image(self):
        if not hasattr(self, "trained_model"):
            self.status_label[
                "text"
            ] = "Model not loaded. Load or train the model first."
            return

        if image_path := filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg")]
        ):
            prediction = predict_image(
                self.trained_model, image_path, self.model_config
            )
            self.status_label["text"] = f"Prediction: {'Dog' if prediction else 'Cat'}"


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()
