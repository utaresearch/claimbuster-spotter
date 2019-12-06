from sklearn.utils import resample
import numpy as np
import pandas as pd
import pickle
import os
import json
import sys
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from absl import logging
from . import transformations as transf
from models import bert2

sys.path.append('..')
from flags import FLAGS


class XLNetExample():
    def __init__(self, text_a, label, guid, text_b=None):
        self.text_a = text_a
        self.label = label
        self.guid = guid
        self.text_b = text_b


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
        xlen, ylen = len(self.x), len(self.y)
        if xlen != ylen:
            raise ValueError("size of x != size of y ({} != {})".format(xlen, ylen))
        return xlen


class DataLoader:
    def __init__(self, train_data=None, val_data=None, test_data=None):
        assert FLAGS.num_classes == 2 or FLAGS.num_classes == 3

        self.data, self.eval_data, self.vocab = self.load_ext_data(train_data, val_data, test_data) \
            if not FLAGS.use_clef_data else self.load_clef_data()

        if FLAGS.use_clef_data and FLAGS.combine_ours_clef_data:
            ours_data, ours_eval, ours_vocab = self.load_ext_data(train_data, val_data, test_data)

            ours_data = self.convert_3_to_2(ours_data)
            ours_eval = self.convert_3_to_2(ours_eval)

            self.data.x += ours_data.x
            self.data.y += ours_data.y
            self.eval_data.x += ours_eval.x
            self.eval_data.y += ours_eval.y

        if FLAGS.num_classes == 2 and not FLAGS.use_clef_data:
            self.data = self.convert_3_to_2(self.data)
            self.eval_data = self.convert_3_to_2(self.data)

        self.class_weights = self.compute_class_weights()
        logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()
        self.post_process_flags()

    @staticmethod
    def convert_3_to_2(data):
        if FLAGS.alt_two_class_combo:       
            data.y = [(0 if data.y[i] == 0 else 1) for i in range(len(data.y))]
        else:
            data.y = [(1 if data.y[i] == 2 else 0) for i in range(len(data.y))]

        return data

    def compute_class_weights(self):
        ret = compute_class_weight('balanced', [z for z in range(FLAGS.num_classes)], self.data.y)

        if FLAGS.num_classes == 3:
            ret[1] /= 4

        return ret

    def load_training_data(self):
        ret = Dataset(self.data.x, self.data.y, FLAGS.random_state)

        if FLAGS.sklearn_oversample:
            classes = [[] for _ in range(FLAGS.num_classes)]

            for i in range(len(ret.x)):
                classes[ret.y[i]].append(ret.x[i])

            if FLAGS.num_classes == 3:
                maj_len = len(classes[2])
                classes[0] = resample(classes[0], n_samples=int(maj_len * 2.75), random_state=FLAGS.random_state)
                classes[1] = resample(classes[1], n_samples=int(maj_len * 0.90), random_state=FLAGS.random_state)
                classes[2] = resample(classes[2], n_samples=int(maj_len * 1.50), random_state=FLAGS.random_state)
            else:
                pass
                # maj_len = len(classes[0])
                # # classes[0] = resample(classes[0], n_samples=int(maj_len), random_state=FLAGS.random_state)
                # classes[1] = resample(classes[1], n_samples=int(maj_len * 0.40), random_state=FLAGS.random_state)

            ret = Dataset([], [], random_state=FLAGS.random_state)
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

        ret.shuffle()

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
    def load_clef_data():
        if not os.path.isfile(FLAGS.prc_clef_loc):
            FLAGS.refresh_data = True

        def read_from_file(loc):
            df = pd.read_csv(loc)
            ret_txt, ret_lab = [row['text'] for idx, row in df.iterrows()], [row['label'] for idx, row in df.iterrows()]
            return ret_txt, ret_lab

        if FLAGS.refresh_data:
            train_txt, train_lab = read_from_file(FLAGS.raw_clef_train_loc)
            eval_txt, eval_lab = read_from_file(FLAGS.raw_clef_test_loc)

            train_features, eval_features = DataLoader.process_text_for_transformers(train_txt, eval_txt,
                                                                                     train_lab, eval_lab)

            logging.info('Loading preprocessing dependencies')
            transf.load_dependencies()
            vocab = None

            logging.info('Processing train data')
            train_txt, train_pos, train_sent = transf.process_dataset(train_txt)
            logging.info('Processing eval data')
            eval_txt, eval_pos, eval_sent = transf.process_dataset(eval_txt)

            train_data = Dataset(list(zip(train_features, train_pos, train_sent)), train_lab,
                                 random_state=FLAGS.random_state)
            eval_data = Dataset(list(zip(eval_features, eval_pos, eval_sent)), eval_lab,
                                random_state=FLAGS.random_state)

            with open(FLAGS.prc_clef_loc, 'wb') as f:
                pickle.dump((train_data, eval_data, vocab), f)
            logging.info('Refreshed data, successfully dumped at {}'.format(FLAGS.prc_clef_loc))
        else:
            logging.info('Restoring data from {}'.format(FLAGS.prc_clef_loc))
            with open(FLAGS.prc_clef_loc, 'rb') as f:
                train_data, eval_data, vocab = pickle.load(f)

        return train_data, eval_data, vocab

    @staticmethod
    def load_ext_data(train_data_in, val_data_in, test_data_in):
        train_data, test_data, vocab = None, None, None

        data_loc = FLAGS.prc_data_loc[:-7] + '_{}'.format('xlnet' if FLAGS.tfm_type == 0 else 'bert') + '.pickle'

        if (train_data_in is not None and val_data_in is not None and test_data_in is not None) or \
           (not os.path.isfile(data_loc)):
            FLAGS.refresh_data = True

        if FLAGS.refresh_data:
            train_data = DataLoader.parse_json(FLAGS.raw_data_loc) if train_data_in is None else train_data_in
            dj_eval_data = DataLoader.parse_json(FLAGS.raw_dj_eval_loc) if test_data_in is None else test_data_in

            train_txt = [z[0] for z in train_data]
            eval_txt = [z[0] for z in dj_eval_data]

            train_lab = [z[1] + 1 for z in train_data]
            eval_lab = [z[1] + 1 for z in dj_eval_data]

            logging.info('Loading preprocessing dependencies')
            transf.load_dependencies()

            logging.info('Processing train data')
            train_txt, train_pos, train_sent = transf.process_dataset(train_txt)
            logging.info('Processing eval data')
            eval_txt, eval_pos, eval_sent = transf.process_dataset(eval_txt)

            train_features, eval_features = DataLoader.process_text_for_transformers(train_txt, eval_txt)

            train_data = Dataset(list(zip(train_features, train_pos, train_sent)), train_lab,
                                 random_state=FLAGS.random_state)
            eval_data = Dataset(list(zip(eval_features, eval_pos, eval_sent)), eval_lab,
                                random_state=FLAGS.random_state)

            with open(data_loc, 'wb') as f:
                pickle.dump((train_data, eval_data, vocab), f)
            logging.info('Refreshed data, successfully dumped at {}'.format(data_loc))
        else:
            logging.info('Restoring data from {}'.format(data_loc))
            with open(data_loc, 'rb') as f:
                train_data, eval_data, vocab = pickle.load(f)

        return train_data, eval_data, vocab

    @staticmethod
    def process_text_for_transformers(train_txt, eval_txt):
        vocab_file = os.path.join(FLAGS.bert_model_loc, "vocab.txt")
        tokenizer = bert2.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)

        train_txt = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in train_txt]
        eval_txt = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in eval_txt]

        return train_txt, eval_txt

    @staticmethod
    def parse_json(json_loc):
        with open(json_loc) as f:
            temp_data = json.load(f)

        dl = []
        labels = [0, 0, 0]

        for el in temp_data:
            lab = int(el["label"])
            txt = el["text"]

            labels[lab + 1] += 1
            dl.append([txt, lab])

        print('{}: {}'.format(json_loc, labels))
        return dl
