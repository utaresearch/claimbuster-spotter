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
    def __init__(self):
        assert FLAGS.num_classes == 2 or FLAGS.num_classes == 3

        self.data, self.eval_data, self.vocab = self.load_external_raw()
        if FLAGS.num_classes == 2:
            self.conv_3_to_2()

        self.class_weights = self.compute_class_weights()
        tf.logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()
        self.post_process_flags()

    def conv_3_to_2(self):
        self.data.y = [(1 if self.data.y[i] == 2 else 0) for i in range(len(self.data.y))]
        self.eval_data.y = [(1 if self.eval_data.y[i] == 2 else 0) for i in range(len(self.eval_data.y))]

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

    def load_testing_data(self):
        ret = Dataset([], [], FLAGS.random_state)

        for i in range(FLAGS.test_examples):
            ret.x.append(self.eval_data.x[i])
            ret.y.append(self.eval_data.y[i])

        return ret

    def post_process_flags(self):
        FLAGS.train_examples = self.data.get_length()
        FLAGS.test_examples = self.eval_data.get_length()
        FLAGS.total_examples = FLAGS.train_examples + FLAGS.test_examples

    @staticmethod
    def load_external_raw():
        train_data = DataLoader.parse_json(FLAGS.raw_data_loc)
        dj_eval_data = DataLoader.parse_json(FLAGS.raw_dj_eval_loc)

        train_txt = [z[0] for z in train_data]
        eval_txt = [z[0] for z in train_data]
        train_lab = [z[1] for z in train_data]
        eval_lab = [z[1] for z in train_data]

        tf.logging.info('Loading preprocessing dependencies')
        transf.load_dependencies()

        tf.logging.info('Processing train data')
        for i in tqdm(range(len(train_data))):
            el = train_data[i]
            el[0] = (transf.process_sentence_ner_spacy(el[0]) if FLAGS.ner_spacy else el[0])
            el[0] = transf.expand_contractions(el[0].lower())

        tf.logging.info('Processing eval data')
        for i in tqdm(range(len(dj_eval_data))):
            el = dj_eval_data[i]
            el[0] = (transf.process_sentence_ner_spacy(el[0]) if FLAGS.ner_spacy else el[0])
            el[0] = transf.expand_contractions(el[0].lower())

        tokenizer = Tokenizer()

        tokenizer.fit_on_texts(np.concatenate((train_txt, eval_txt)))
        train_seq = tokenizer.texts_to_sequences(train_txt)
        eval_seq = tokenizer.texts_to_sequences(eval_txt)

        train_data = Dataset(train_seq, train_lab, random_state=FLAGS.random_state)
        eval_data = Dataset(eval_seq, eval_lab, random_state=FLAGS.random_state)
        vocab = tokenizer.word_index

        return train_data, eval_data, vocab

    @staticmethod
    def parse_json(json_loc):
        with open(json_loc) as f:
            temp_data = json.load(f)

        dl = []
        labels = [0, 0, 0]

        for el in temp_data:
            lab = int(el["label"]) + 1
            txt = el["text"]

            labels[lab] += 1
            dl.append([txt, lab])

        print('{}: {}'.format(json_loc, labels))
        return dl
