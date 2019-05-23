import tensorflow as tf
import sys
sys.path.append('..')
from flags import FLAGS


class RecurrentModel:
    def __init__(self):
        self.lstm_layers = FLAGS.rnn_num_layers
        self.num_hidden = FLAGS.rnn_cell_size

    def construct_model(self, x):
        y = self.build_lstm(x)
        return y, self.compute_loss(y)

    def build_lstm(self, x):
        embed = tf.contrib.layers.embed_sequence(x, vocab_size=int(1e6), embed_dim=FLAGS.embedding_dims)
        lstm = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm() for _ in range(self.lstm_layers)])

        output, state = tf.nn.dynamic_rnn(cell=lstm, inputs=embed, dtype=tf.float32)
        return output[-1]

    def get_lstm(self):
        return tf.nn.rnn_cell.LSTMCell(self.num_hidden)

    def compute_loss(self, y):
        print('foo')
        return 'fo'
