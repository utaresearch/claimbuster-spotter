import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.utils import resample
import numpy as np
import pickle
import math
from tqdm import tqdm
import json
import sys
from . import transformations as transf
sys.path.append('..')
from flags import FLAGS
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

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

    def get_length(self):
        if len(self.x) != len(self.y):
            raise ValueError("size of x != size of y ({} != {})".format(len(self.x), len(self.y)))
        return len(self.x)


class DataLoader:
    def __init__(self, custom_prc_data_loc=None, custom_vocab_loc=None, ver='train'):
        assert (custom_prc_data_loc is None and custom_prc_data_loc is None) or \
               (custom_prc_data_loc is not None and custom_prc_data_loc is not None)
        assert FLAGS.num_classes == 2 or FLAGS.num_classes == 3

        # self.data = self.load_external() if (not custom_prc_data_loc and not custom_vocab_loc) else \
        #     self.load_external_custom(custom_prc_data_loc, custom_vocab_loc)
        self.data = self.load_external_raw()
        if FLAGS.num_classes == 2:
            self.conv_3_to_2()

        self.class_weights = self.compute_class_weights()
        tf.logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()

        if (custom_prc_data_loc is None and custom_vocab_loc is None) or ver == 'eval':
            self.post_process_flags()
        else:
            FLAGS.total_examples += self.data.get_length()
            FLAGS.validation_examples += self.data.get_length()

    def conv_3_to_2(self):
        self.data.y = [(1 if self.data.y[i] == 2 else 0) for i in range(len(self.data.y))]

    def compute_class_weights(self):
        return compute_class_weight('balanced', [z for z in range(FLAGS.num_classes)], self.data.y)

    def load_training_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.train_examples):
            ret.x.append(self.data.x[i])
            ret.y.append(self.data.y[i])

        if FLAGS.sklearn_oversample:
            classes = [[] for _ in range(FLAGS.num_classes)]

            for i in range(len(ret.x)):
                classes[ret.y[i]].append(ret.x[i])

            if FLAGS.num_classes == 3:
                maj_len = len(classes[2])
                classes[0] = resample(classes[0], n_samples=int(maj_len * 2.50), random_state=FLAGS.random_state)
                classes[1] = resample(classes[1], n_samples=int(maj_len * 0.90), random_state=FLAGS.random_state)
                classes[2] = resample(classes[2], n_samples=int(maj_len * 1.40), random_state=FLAGS.random_state)
            else:
                maj_len = len(classes[0])
                # classes[0] = resample(classes[0], n_samples=int(maj_len), random_state=FLAGS.random_state)
                classes[1] = resample(classes[1], n_samples=int(maj_len * 0.40), random_state=FLAGS.random_state)

            ret.x, ret.y = [], []
            del self.data.x[:FLAGS.train_examples]
            del self.data.y[:FLAGS.train_examples]

            for lab in range(len(classes)):
                for inp_x in classes[lab]:
                    ret.x.append(inp_x)
                    ret.y.append(lab)
                    self.data.x.insert(0, inp_x)
                    self.data.y.insert(0, lab)

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

    def load_separate_validation(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.validation_examples):
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
    def load_external_raw():
        train_data = DataLoader.parse_json(FLAGS.raw_json_loc)
        dj_eval_loc = DataLoader.parse_json(FLAGS.raw_dj_eval_loc)

        train_txt = [z[0] for z in train_data]
        eval_txt = [z[0] for z in train_data]

        transf.load_dependencies()

        for el in train_data:
            el[0] = (transf.process_sentence_ner_spacy(el[0]) if FLAGS.ner_spacy else el[0])
            el[0] = transf.exp_contractions(el[0].lower())
        for el in dj_eval_loc:
            el[0] = (transf.process_sentence_ner_spacy(el[0]) if FLAGS.ner_spacy else el[0])
            el[0] = transf.exp_contractions(el[0].lower())

        tokenizer = Tokenizer()

        tokenizer.fit_on_texts(np.concatenate((train_txt, eval_txt)))
        train_seq = tokenizer.texts_to_sequences(train_txt)
        eval_seq = tokenizer.texts_to_sequences(eval_txt)

        print(train_seq)
        print(eval_seq)

        exit()

        return train_data, dj_eval_loc, train_vocab,

    @staticmethod
    def process_dataset(parsed_data):
        for el in parsed_data:
            el[0] = (transf.process_sentence_ner_spacy(el[0]) if FLAGS.ner_spacy else el[0])
            el[0] = transf.exp_contractions(el[0].lower())

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([z[0] for z in parsed_data])
        sequences = tokenizer.texts_to_sequences(texts)

    @staticmethod
    def parse_json(json_loc):
        with open(json_loc) as f:
            temp_data = json.load(f)

        dl = []
        labels = [0, 0, 0]

        for el in temp_data:
            lab = int(el["label"]) + 1
            txt = f["text"]

            labels[lab] += 1
            dl.append((txt, lab))

        print(labels)
        return dl

    @staticmethod
    def get_default_vocab():
        with open(FLAGS.vocab_loc, 'rb') as f:
            return pickle.load(f)