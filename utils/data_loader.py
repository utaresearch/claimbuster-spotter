import tensorflow as tf
import numpy as np
import pickle
import math
import sys
import os
sys.path.append('..')
from flags import FLAGS
from sklearn.utils import shuffle


class Dataset:
    x = []
    y = []

    def __init__(self, x, y, random_state):
        self.x = x
        self.y = y
        self.random_state = random_state
        self.shuffle()

    def shuffle(self):
        self.x, self.y = shuffle(self.x, self.y, random_state=self.random_state)

    def get_length(self):
        if len(self.x) != len(self.y):
            raise ValueError("size of x != size of y ({} != {})".format(len(self.x), len(self.y)))
        return len(self.x)


class DataLoader:
    def __init__(self, custom_prc_data_loc=None, custom_vocab_loc=None):
        assert (custom_prc_data_loc is None and custom_prc_data_loc is None) or \
               (custom_prc_data_loc is not None and custom_prc_data_loc is not None)
        self.data = self.load_external() if (not custom_prc_data_loc and not custom_vocab_loc) else \
            self.load_external_custom(custom_prc_data_loc, custom_vocab_loc)
        self.data.shuffle()
        self.post_process_flags()

    def load_training_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.train_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    def load_validation_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.train_examples, FLAGS.train_examples + FLAGS.validation_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    def load_testing_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.train_examples + FLAGS.validation_examples, FLAGS.total_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    def load_all_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.total_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    def post_process_flags(self):
        FLAGS.total_examples = self.data.get_length()
        FLAGS.train_examples = int(math.ceil(float(FLAGS.total_examples) * FLAGS.train_pct))
        FLAGS.test_examples = FLAGS.total_examples - FLAGS.train_examples
        FLAGS.validation_examples = int(math.floor(float(FLAGS.total_examples) * FLAGS.validation_pct))
        FLAGS.train_examples = FLAGS.train_examples - FLAGS.validation_examples

    @staticmethod
    def load_external():
        with open(FLAGS.prc_data_loc, 'rb') as f:
            data = pickle.load(f)
        with open(FLAGS.vocab_loc, 'rb') as f:
            vc = [x[0] for x in pickle.load(f)]

        return Dataset([[vc.index(ch) for ch in x[1].split(' ')] for x in data],
                       [int(x[0]) + 1 for x in data], FLAGS.random_state)

    @staticmethod
    def load_external_custom(custom_prc_data_loc, custom_vocab_loc):
        with open(custom_prc_data_loc, 'rb') as f:
            data = pickle.load(f)
        with open(custom_vocab_loc, 'rb') as f:
            vc = [x[0] for x in pickle.load(f)]

        with tf.Session() as sess:
            print(DataLoader.load_embedding_dict(sess))
            embed_dict = DataLoader.load_embedding_dict(sess).eval()
            assert (embed_dict != 0)

        print(embed_dict)
        exit()

        return Dataset([[vc.index(ch) for ch in x[1].split(' ')] for x in data],
                       [int(x[0]) + 1 for x in data], FLAGS.random_state)

    @staticmethod
    def load_embedding_dict(sess):
        target_file = os.path.join(FLAGS.output_dir, "embedding_matrix_tf.ckpt")
        tf.logging.info("Attempting to restore embedding matrix backup from {}...".format(target_file))

        var_to_return = tf.Variable(0, dtype=tf.float32)

        try:
            saver = tf.train.Saver({"var_to_return": var_to_return})
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, target_file)
            return var_to_return.eval()
        except:
            print('faileddddd')
            sess.run(tf.global_variables_initializer())
            return var_to_return.eval()
