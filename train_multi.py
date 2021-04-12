import tensorflow as tf
from sklearn.datasets import make_classification


X_train, y_train = make_classification(n_samples=1000, n_features=3, n_classes=4, n_redundant=0, n_informative=3)

nn = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu)
])
nn.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
nn.fit(X_train, y_train, epochs=10, batch_size=500, verbose=False)

tf.keras.models.save_model(nn, 'models/multi')