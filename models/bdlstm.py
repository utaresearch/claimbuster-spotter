import tensorflow as tf
from embedding import Embedding
import sys
sys.path.append('..')
from flags import FLAGS


class RecurrentModel:
    def __init__(self):
        pass

    def construct_model(self, x, y):
        yhat = self.build_lstm(x)
        return yhat, self.compute_loss(y, yhat)

    def build_lstm(self, x):
        # embed = tf.contrib.layers.embed_sequence(x, vocab_size=int(100), embed_dim=FLAGS.embedding_dims)

        vocab_size = 0
        vocab_freqs = 0
        vocab_list = 0
        embed = Embedding(vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
                          FLAGS.keep_prob, FLAGS.keep_prob_emb, vocab_freqs, vocab_list,
                          FLAGS.w2v_loc, FLAGS.transfer_learn_w2v, FLAGS.data_dir)
        exit()

        lstm = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm() for _ in range(FLAGS.rnn_num_layers)])

        output, state = tf.nn.dynamic_rnn(cell=lstm, inputs=embed, dtype=tf.float32)

        add_weight = tf.get_variable('post_lstm_weight', shape=(FLAGS.rnn_cell_size, FLAGS.num_classes),
                                     initializer=tf.contrib.layers.xavier_initializer())
        add_bias = tf.get_variable('post_lstm_bias', shape=FLAGS.num_classes,
                                   initializer=tf.contrib.layers.xavier_initializer())

        return tf.matmul(output[-1], add_weight) + add_bias

    @staticmethod
    def get_lstm():
        return tf.nn.rnn_cell.LSTMCell(FLAGS.rnn_cell_size)

    @staticmethod
    def compute_loss(y, yhat):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat)
