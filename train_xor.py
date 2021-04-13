import tensorflow as tf
import numpy as np


X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([
    0,
    1,
    1,
    0
])

nn = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
])
nn.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5)
)
nn.fit(X_train, y_train, epochs=2000, verbose=False)

tf.keras.models.save_model(nn, 'models/xor')