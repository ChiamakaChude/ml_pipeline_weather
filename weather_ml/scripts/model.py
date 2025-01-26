import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from config.config import model_path

def modelling(data, save_model):
    
    # Prepare the target and features
    X = data.drop(['traffic_speed'], axis=1).values
    y = data['traffic_speed'].values

    # Time-based train-test split (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    
    model = Sequential()

    # LSTM layer
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Dense output layer (we are predicting a continuous value, traffic speed)
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    
    if save_model:
        model.save(model_path)
        print("Model saved as 'traffic_speed_model.h5'")

    # Optionally, return the model and history for further analysis
    return model, history
