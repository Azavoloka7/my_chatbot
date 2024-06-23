import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def train_model(model, train_x, train_y, epochs=200, batch_size=5, verbose=1):
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.save('../models/chatbot_model.keras')
    print("Model training complete and saved as 'models/chatbot_model.keras'")
