import tensorflow as tf
import numpy as np
import pickle
import sys
import os
from gensim.models import KeyedVectors
sys.path.append('..')
from flags import FLAGS


class Embedding:
    def __init__(self):
        self.vocab_list, self.vocab_freqs = self.get_vocab()
        self.embed_shape = (len(self.vocab_list), FLAGS.embedding_dims)

    def init_embeddings(self, sess):
        embed = tf.Variable(np.zeros(self.embed_shape))
        w2v = tf.placeholder(tf.float32, shape=self.embed_shape)
        embed_init_op = embed.assign(w2v)

        sess.run(embed_init_op, feed_dict={
            w2v: self.create_embedding_matrix()
        })

    def create_embedding_matrix(self):
        retrieve_val = self.retrieve_embedding_matrix()
        equal_to_init = (retrieve_val == np.zeros(self.embed_shape, dtype=np.dtype('float32'))).all()

        if equal_to_init:
            tf.logging.info("Embedding matrix not found.")
        else:
            tf.logging.info("Embedding matrix found.")
            return retrieve_val

        embedding_matrix = np.zeros(self.embed_shape, dtype=np.dtype('float32'))

        tf.logging.info("Loading word2vec model...")
        model = KeyedVectors.load_word2vec_format(FLAGS.w2v_loc, binary=True)
        tf.logging.info("Model loaded.")

        idx, fail_cnt, fail_words = 0, 0, []
        for word in self.vocab_list:
            try:
                embedding_vector = model[word]
                embedding_matrix[idx] = embedding_vector
            except:
                fail_cnt = fail_cnt + 1
                fail_words.append(word)
            idx = idx + 1

        fail_words.sort()
        tf.logging.info(str(fail_cnt) + " out of " + str(idx) +
                        " strings were not found in word2vec and were defaulted to zeroes.")
        tf.logging.info(fail_words)

        var_to_return = tf.Variable(embedding_matrix)

        # Save variable
        tf.logging.info("Saving generated embedding matrix...")
        saver = tf.train.Saver({"var_to_return": var_to_return})

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            save_path = saver.save(sess, os.path.join(FLAGS.data_dir, "/embedding_matrix_tf.ckpt"))
            tf.logging.info("Model saved in path: " + save_path)

        return embedding_matrix

    def retrieve_embedding_matrix(self):
        target_file = os.path.join(FLAGS.data_dir,"/embedding_matrix_tf.ckpt")
        tf.logging.info("Attempting to restore embedding matrix backup from {}...".format(target_file))

        """Useful debugging tool"""
        # from tensorflow.python.tools import inspect_checkpoint as chkp
        # chkp.print_tensors_in_checkpoint_file(target_file, tensor_name='', all_tensors=True)

        var_to_return = tf.Variable(np.zeros(self.embed_shape, dtype=np.dtype('float32')))

        try:
            saver = tf.train.Saver({"var_to_return": var_to_return})
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, target_file)
                return var_to_return.eval()
        except:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                return var_to_return.eval()

    @staticmethod
    def get_vocab():
        with open(FLAGS.vocab_path, 'rb') as f:
            data = pickle.load(f)
        return [x[0] for x in data], [x[1] for x in data]