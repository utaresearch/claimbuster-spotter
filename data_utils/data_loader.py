import pickle
import sys
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
    def __init__(self):
        self.data = self.load_external()
        self.data.shuffle()
        self.post_process_flags()

    def load_training_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.training_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    def load_validation_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.training_examples, FLAGS.training_examples + FLAGS.validation_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    def load_testing_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.training_examples + FLAGS.validation_examples, FLAGS.total_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        return ret

    @staticmethod
    def post_process_flags():
        print('foo')

    @staticmethod
    def load_external():
        with open(FLAGS.data_pkl_loc, 'rb') as f:
            data = pickle.load(f)

        print(data)
        exit()
