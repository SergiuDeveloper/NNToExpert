import tensorflow as tf
from sklearn.datasets import make_classification


X_train, y_train = make_classification(n_samples=1000, n_features=3, n_classes=5, n_clusters_per_class=1, n_redundant=0, n_informative=3)

nn = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu)
])
nn.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
nn.fit(X_train, y_train, epochs=100, batch_size=5, verbose=False)

tf.keras.models.save_model(nn, 'models/multi')