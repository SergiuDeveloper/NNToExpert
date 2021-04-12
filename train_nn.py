import tensorflow as tf
import numpy as np


X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([
    [0],
    [0],
    [0],
    [1]
])

nn = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
])
nn.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
)
nn.fit(X_train, y_train, epochs=30, batch_size=1, verbose=False)

tf.keras.models.save_model(nn, 'models/and')