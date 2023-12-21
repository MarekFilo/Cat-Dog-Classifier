# Cat-Dog Classifier

The Cat-Dog Classifier is a simple application that uses a Convolutional Neural Network (CNN) to classify whether an input image contains a cat or a dog.

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MarekFilo/Cat-Dog-Classifier.git
   ```

2. **Navigate to the project directory**:

    ```bash
    cd Cat-Dog-Classifier
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run main file**:
    ```bash
    python src/main.py
    ```
Upon running the main file, a window will appear allowing you to load or train a model and make predictions.

## Project Structure
- src/model/: Contains the neural network for the Cat-Dog Classifier model.
- src/config/: Necessery configuration files
- src/: train.py, predict.py and main.py
- data_example/: Example directory structure for training and testing datasets.



### Note: 
The 'data_example/ directory' is provided as a reference and should be renamed to 'data' as original dataset was too large or change the path `TrainConfig` class located in `src/config/train_config`. You should replace it with the actual training and testing datasets. Data used for training can be found here [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data) by [tongpython](https://www.kaggle.com/tongpython) 

## Model Performance and Potential Improvements

Due to limited computing resources, the model may not achieve optimal performance, and there is room for improvement.

### Suggestions for Model Improvement:

1. **Adjust Model Configuration:** Experiment with different configurations in the `ModelConfig` class, located in `src/config/model_config` , such as changing the input shape, batch size, number of epochs, and dropout rate. You can also try adding new options in the `ModelConfig`, more convolutional layers or increasing the complexity of the neural network.

2. **Hyperparameter Tuning:** Fine-tune hyperparameters like learning rate, dropout rate, or regularization strength to find the best combination for your specific dataset.

3. **Data Augmentation:** Apply data augmentation techniques to artificially increase the size of training dataset and enhance the model's ability to generalize to unseen data.

4. **Batch Normalization:** Introduce Batch Normalization layers to normalize the activations between layers, potentially improving training speed and model convergence.

