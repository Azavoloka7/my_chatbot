import numpy as np
from model_training import create_model, train_model
import os

if __name__ == "__main__":
    # Check if training data exists
    if not os.path.exists('../data/train_x.npy') or not os.path.exists('../data/train_y.npy'):
        raise FileNotFoundError("Training data not found. Please run data_preparation.py first.")
    
    # Load preprocessed data
    train_x = np.load('../data/train_x.npy')
    train_y = np.load('../data/train_y.npy')

    # Create and train the model
    input_shape = len(train_x[0])
    output_shape = len(train_y[0])
    model = create_model(input_shape, output_shape)
    
    train_model(model, train_x, train_y, epochs=200, batch_size=5, verbose=1)
