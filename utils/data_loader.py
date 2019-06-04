import pickle
import math
import sys
sys.path.append('..')
from flags import FLAGS
from sklearn.utils import shuffle


fail_cnt = 0


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
        FLAGS.validation_examples = int(math.floor(float(FLAGS.total_examples) * FLAGS.validation_pct))
        FLAGS.test_examples = FLAGS.total_examples - FLAGS.train_examples - FLAGS.validation_examples

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

        default_vocab = DataLoader.get_default_vocab()

        def vocab_idx(ch):
            global fail_cnt
            try:
                return default_vocab.index(ch)
            except:
                fail_cnt += 1
                return -1

        print('{} out of {} words were not found are defaulted to -1.'.format(fail_cnt, len(vc)))

        return Dataset([[vocab_idx(ch) for ch in x[1].split(' ')] for x in data],
                       [int(x[0]) + 1 for x in data], FLAGS.random_state)

    @staticmethod
    def get_default_vocab():
        with open(FLAGS.vocab_loc, 'rb') as f:
            return [x[0] for x in pickle.load(f)]
