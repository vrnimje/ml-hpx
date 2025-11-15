from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from ml_hpx import SGD, Layer, NeuralNetwork

# Dataset - XOR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# HPX NN
layers = [Layer(4, 2, "relu"), Layer(1, 4, "sigmoid")]
optimizer = SGD(0.1)
nn = NeuralNetwork(layers, optimizer)

start = perf_counter()
nn.fit(X.tolist(), y.tolist(), 1000)  # 5000 epochs
end = perf_counter()
print(f"HPX NN training time: {end - start:.6f} sec")

preds = nn.predict(X.tolist())
print("HPX Predictions:", preds)

# TensorFlow NN
model = Sequential(
    [
        Input(shape=(2,)),  # explicitly declare input shape
        Dense(4, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

start = perf_counter()
model.fit(X, y, epochs=1000, verbose=0)
end = perf_counter()
print(f"TensorFlow training time: {end - start:.6f} sec")

preds_tf = model.predict(X)
print("TF Predictions:", preds_tf.tolist())
