from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import numpy as np
import pickle
import math
import sys
sys.path.append('..')
from flags import FLAGS
from sklearn.utils import shuffle

fail_words = set()


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
        # c0, c1, c2 = [], [], []
        #
        # for i in range(len(self.x)):
        #     if self.y[i] == 0:
        #         c0.append((self.x[i], self.y[i]))
        #     elif self.y[i] == 1:
        #         c1.append((self.x[i], self.y[i]))
        #     elif self.y[i] == 2:
        #         c2.append((self.x[i], self.y[i]))
        #
        # def shuffle_util(cc):
        #     t1, t2 = shuffle([x[0] for x in cc], [x[1] for x in cc], random_state=self.random_state)
        #     return list(zip(t1, t2))
        #
        # def get_target_num(tot_ex):
        #     train_ex = int(math.ceil(float(tot_ex) * FLAGS.train_pct))
        #     val_ex = int(math.floor(float(tot_ex) * FLAGS.validation_pct))
        #     test_ex = tot_ex - train_ex - val_ex
        #     return train_ex, val_ex, test_ex
        #
        # new_x, new_y = [], []
        # all_ars = [shuffle_util(c0), shuffle_util(c1), shuffle_util(c2)]
        # for ar in all_ars:
        #     temp_x = [f[0] for f in ar]
        #     temp_y = [f[1] for f in ar]
        #
        #     trex, vex, teex = get_target_num(len(ar))
        #     np.concatenate((new_x, temp_x[:trex]))
        #     np.concatenate((new_x, temp_x[trex:trex + vex]))
        #     np.concatenate((new_x, temp_x[trex + vex:len(temp_x)]))
        #
        #     np.concatenate((new_y, temp_y[:trex]))
        #     np.concatenate((new_y, temp_y[trex:trex + vex]))
        #     np.concatenate((new_y, temp_y[trex + vex:len(temp_y)]))

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

        if FLAGS.smote_synthetic:
            sm = SMOTE(random_state=FLAGS.random_state, ratio=1.0)
            ret.x, ret.y = sm.fit_sample(ret.x, ret.y)
        elif FLAGS.sklearn_oversample:
            c0, c1, c2 = [], [], []

            for i in range(len(ret.x)):
                if ret.y[i] == 0:
                    c0.append((ret.x[i], ret.y[i]))
                elif ret.y[i] == 1:
                    c1.append((ret.x[i], ret.y[i]))
                elif ret.y[i] == 2:
                    c2.append((ret.x[i], ret.y[i]))

            # maj_len = max(len(c0), len(c1), len(c2))
            maj_len = len(c2)
            c0 = resample(c0, n_samples=int(maj_len * 2 * 2.5), random_state=FLAGS.random_state)
            c1 = resample(c1, n_samples=int(maj_len * 2), random_state=FLAGS.random_state)
            c2 = resample(c2, n_samples=int(maj_len * 2 * 1.25), random_state=FLAGS.random_state)

            ret.x, ret.y = [], []
            del self.data.x[:FLAGS.train_examples]
            del self.data.y[:FLAGS.train_examples]
            for el in np.concatenate((c0, c1, c2)):
                ret.x.append(el[0])
                ret.y.append(el[1])
                self.data.x.insert(0, el[0])
                self.data.y.insert(0, el[1])

            FLAGS.total_examples += ret.get_length() - FLAGS.train_examples
            FLAGS.train_examples = ret.get_length()

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
        FLAGS.validation_examples = int(math.floor(float(FLAGS.total_examples) * FLAGS.validation_pct))
        FLAGS.test_examples = FLAGS.total_examples - FLAGS.train_examples - FLAGS.validation_examples

    @staticmethod
    def load_external():
        with open(FLAGS.prc_data_loc, 'rb') as f:
            data = pickle.load(f)
        with open(FLAGS.vocab_loc, 'rb') as f:
            vc = pickle.load(f)

        return Dataset([[vc.index(ch) for ch in x[1].split(' ')] for x in data],
                       [int(x[0]) + 1 for x in data], FLAGS.random_state)

    @staticmethod
    def load_external_custom(custom_prc_data_loc, custom_vocab_loc):
        with open(custom_prc_data_loc, 'rb') as f:
            data = pickle.load(f)
        with open(custom_vocab_loc, 'rb') as f:
            vc = pickle.load(f)

        default_vocab = DataLoader.get_default_vocab()

        def vocab_idx(ch):
            global fail_words
            try:
                return default_vocab.index(ch)
            except:
                fail_words.add(ch)
                return -1

        ret = Dataset([[vocab_idx(ch) for ch in x[1].split(' ')] for x in data],
                      [int(x[0]) + 1 for x in data], FLAGS.random_state)

        print(fail_words)
        print('{} out of {} words were not found are defaulted to -1.'.format(len(fail_words), len(vc)))
        return ret

    @staticmethod
    def get_default_vocab():
        with open(FLAGS.vocab_loc, 'rb') as f:
            return pickle.load(f)