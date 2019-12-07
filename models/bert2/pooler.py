import tensorflow as tf

K = tf.keras
L = K.layers


class PoolerLayer(L.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(PoolerLayer, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dense_layer = L.Dense(hidden_size, name='dense')

    def call(self, x):
        return self.dense_layer(x)
