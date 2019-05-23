import tensorflow as tf

from gensim.models import KeyedVectors
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops

K = tf.keras


class Embedding(K.layers.Layer):
    def __init__(self, vocab_size,
                 embedding_dim, normalize=False,
                 vocab_freqs=None, vocab_list=None,
                 keep_prob=1., w2v_loc=None,
                 transfer_learn_w2v=False, data_dir=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.normalized = normalize
        self.keep_prob = keep_prob
        self.vocab_freqs = vocab_freqs
        self.vocab_list = vocab_list
        self.w2v_loc = w2v_loc
        self.transfer_learn_w2v = transfer_learn_w2v
        self.data_dir = data_dir

        tern = " " if self.transfer_learn_w2v else " not "
        tf.logging.info("w2v embeddings will" + tern + "be trained on.")

        self.embedding_matrix_tf = self.create_embedding_matrix()

        if normalize:
            assert vocab_freqs is not None
            self.vocab_freqs = tf.constant(
                vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))

        super(Embedding, self).__init__()

    def return_initialized_value(self):
        var = self.embedding_matrix_tf
        with ops.init_scope():
            return control_flow_ops.cond(state_ops.is_variable_initialized(var),
                                         var.read_value,
                                         lambda: var.initial_value)

    def create_embedding_matrix(self):
        retrieve_val = self.retrieve_embedding_matrix()
        equal_to_init = tf.reduce_all(tf.equal(retrieve_val, tf.Variable(
            np.zeros((self.vocab_size, self.embedding_dim), dtype=np.dtype('float32')))))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(retrieve_val.eval())
            b = sess.run(equal_to_init)
            if not b:
                tf.logging.info("Embedding matrix found.")
                sess.close()
                return retrieve_val
            else:
                tf.logging.info("Embedding matrix not found.")
            sess.close()

        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.dtype('float32'))
        tf.logging.info("Loading word2vec model...")
        model = KeyedVectors.load_word2vec_format(self.w2v_loc, binary=True)
        tf.logging.info("Model loaded.")
        idx = 0
        fail_cnt = 0
        fail_words = []
        tot_cnt = 0
        for word in self.vocab_list:
            try:
                embedding_vector = model[word]
                embedding_matrix[idx] = embedding_vector
            except:
                fail_cnt = fail_cnt + 1
                fail_words.append(word)
            tot_cnt = tot_cnt + 1
            idx = idx + 1
        fail_words.sort()
        tf.logging.info(str(fail_cnt) + " out of " + str(
            tot_cnt) + " strings were not found in word2vec and were defaulted to zeroes.")
        tf.logging.info(fail_words)

        var_to_return = tf.Variable(embedding_matrix)

        # Save variable
        tf.logging.info("Saving generated embedding matrix...")
        saver = tf.train.Saver({"var_to_return": var_to_return})

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            save_path = saver.save(sess, self.data_dir + "/embedding_matrix_tf.ckpt")
            tf.logging.info("Model saved in path: " + save_path)

        return var_to_return

    def retrieve_embedding_matrix(self):
        tf.logging.info("Attempting to restore embedding matrix backup...")
        target_file = self.data_dir + "/embedding_matrix_tf.ckpt"

        """Useful debugging tool"""
        # from tensorflow.python.tools import inspect_checkpoint as chkp
        # chkp.print_tensors_in_checkpoint_file(target_file, tensor_name='', all_tensors=True)

        var_to_return = tf.Variable(np.zeros((self.vocab_size, self.embedding_dim), dtype=np.dtype('float32')))

        try:
            saver = tf.train.Saver({"var_to_return": var_to_return})
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, target_file)
                return tf.Variable(var_to_return.eval())
        except:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                return tf.Variable(var_to_return.eval())

    def call(self, x):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1.:
            shape = embedded.get_shape().as_list()

            # Use same dropout masks at each timestep with specifying noise_shape.
            # This slightly improves performance.
            # Please see https://arxiv.org/abs/1512.05287 for the theoretical
            # explanation.
            embedded = tf.nn.dropout(embedded, self.keep_prob, noise_shape=(shape[0], 1, shape[2]))
        return embedded

    def _normalize(self, emb):
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev
