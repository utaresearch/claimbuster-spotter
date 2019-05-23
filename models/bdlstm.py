import tensorflow as tf
import sys
sys.path.append('..')
from flags import FLAGS


class RecurrentModel:
    def __init__(self):
        self.lstm_layers = FLAGS.rnn_num_layers
        self.num_hidden = FLAGS.rnn_cell_size

    def build_lstm_model(self, x):
        lstm = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm() for _ in range(self.lstm_layers)])
        output, state = tf.nn.static_rnn(cell=lstm, inputs=x, dtype=tf.float32)
        output = output[-1], state[-1]
        return output

    def get_lstm(self):
        return tf.nn.rnn_cell.LSTMCell(self.num_hidden)

    def compute_loss(self):
        print('foo')
