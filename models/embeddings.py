import tensorflow as tf
import numpy as np
import pickle
import sys
import os
from gensim.models import KeyedVectors
sys.path.append('..')
from flags import FLAGS


class Embedding:
    def __init__(self, vocab):
        self.vocab = vocab
        self.embed_shape = (len(self.vocab) + 1, FLAGS.embedding_dims)
        self.embed = None

    def construct_embeddings(self):
        self.embed = tf.Variable(np.zeros(self.embed_shape), dtype=tf.float32, name='embedding',
                                 trainable=FLAGS.train_embed)
        tf.logging.info("Word vectors will{} be trained on".format("" if FLAGS.train_embed else " not"))
        return self.embed

    def init_embeddings(self, sess):
        w2v = tf.placeholder(tf.float32, shape=self.embed_shape)
        embed_init_op = self.embed.assign(w2v)

        sess.run(embed_init_op, feed_dict={
            w2v: self.create_embedding_matrix(sess)
        })

    def create_embedding_matrix(self, sess):
        retrieve_val = self.retrieve_embedding_matrix(sess)
        equal_to_init = (retrieve_val == np.zeros(self.embed_shape, dtype=np.dtype('float32'))).all()

        if equal_to_init:
            tf.logging.info("Embedding matrix not found.")
        else:
            tf.logging.info("Embedding matrix found.")
            return retrieve_val

        embedding_matrix = np.random.normal(loc=0, scale=0.1, size=self.embed_shape).astype(np.float32) \
            if FLAGS.random_init_oov else np.zeros(self.embed_shape, dtype=np.dtype('float32'))

        embedding_matrix[0] = np.zeros(FLAGS.embedding_dims)

        tf.logging.info("Loading {} model...".format('word2vec' if FLAGS.embed_type == 0 else 'glove'))
        model = KeyedVectors.load_word2vec_format(FLAGS.w2v_loc if FLAGS.embed_type == 0 else FLAGS.glove_loc,
                                                  binary=False)
        tf.logging.info("Model loaded.")

        fail_words = []
        for word, idx in self.vocab.items():
            try:
                embedding_vector = model[word]
                embedding_matrix[idx] = embedding_vector
            except Exception:
                fail_words.append(word)

        fail_words.sort()
        tf.logging.info(str(len(fail_words)) + " out of " + str(len(self.vocab)) +
                        " strings were not found and were defaulted.")
        tf.logging.info(fail_words)

        var_to_return = tf.Variable(embedding_matrix)

        # Save variable
        tf.logging.info("Saving generated embedding matrix...")
        saver = tf.train.Saver({"var_to_return": var_to_return})

        sess.run(tf.global_variables_initializer())
        save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "embedding_matrix_tf.ckpt"))
        tf.logging.info("Model saved in path: " + save_path)

        return embedding_matrix

    def retrieve_embedding_matrix(self, sess):
        target_file = os.path.join(FLAGS.output_dir, "embedding_matrix_tf.ckpt")
        tf.logging.info("Attempting to restore embedding matrix backup from {}...".format(target_file))

        var_to_return = tf.Variable(np.zeros(self.embed_shape, dtype=np.dtype('float32')))

        try:
            saver = tf.train.Saver({"var_to_return": var_to_return})
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, target_file)
            return var_to_return.eval()
        except:
            sess.run(tf.global_variables_initializer())
            return var_to_return.eval()

    @staticmethod
    def get_vocab():
        with open(FLAGS.vocab_loc, 'rb') as f:
            return pickle.load(f)
