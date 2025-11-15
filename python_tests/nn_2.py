from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense

from ml_hpx import SGD, Layer, NeuralNetwork

# Dataset
# Non-linear classification problem (10000 samples, 20 features after expansion)
X, y = make_moons(n_samples=10000, noise=0.2, random_state=42)
X = np.hstack([X, np.random.randn(X.shape[0], 18)])  # add 18 noisy features → 20 total
y = y.reshape(-1, 1).astype(np.float32)

X = X.astype(np.float32)

# TensorFlow NN
model = Sequential(
    [
        Input(shape=(20,)),  # Input: 20 features
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

start = perf_counter()
history = model.fit(X, y, epochs=50, batch_size=32, verbose=0)
end = perf_counter()
print(f"TensorFlow training time: {end - start:.6f} sec")

loss, acc = model.evaluate(X, y, verbose=0)
print(f"TF Accuracy: {acc:.4f}, Loss: {loss:.4f}")

preds_tf = model.predict(X[:10], verbose=0)
print("TF Predictions (first 10):", preds_tf.flatten().tolist())

# HPX NN

layers = [Layer(32, 20, "relu"), Layer(16, 32, "relu"), Layer(1, 16, "sigmoid")]
optimizer = SGD(0.05)
nn = NeuralNetwork(layers, optimizer)

start = perf_counter()
nn.fit(X.tolist(), y.tolist(), 50)
end = perf_counter()
print(f"HPX NN training time: {end - start:.6f} sec")

loss, acc = nn.evaluate(X.tolist(), y.tolist())
print(f"HPX NN Accuracy: {acc:.4f}, Loss: {loss:.4f}")

preds_hpx = nn.predict(X[:10].tolist())
print("HPX Predictions (first 10):", preds_hpx)
