from sklearn.utils import resample
import pandas as pd
import pickle
import os
import json
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from absl import logging
from . import transformations as transf
from ..models import bert2
from ..models.custom_albert_tokenization import CustomAlbertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .flags import FLAGS


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
    def __init__(self):
        self.data, self.eval_data, = self.load_ext_data()

        self.class_weights = self.compute_class_weights()
        logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()
        self.post_process_flags()

    def compute_class_weights(self):
        print(self.data.y)
        ret = compute_class_weight('balanced', [z for z in range(FLAGS.cs_num_classes)], self.data.y)

        if FLAGS.cs_num_classes == 3:
            ret[1] /= 4

        return ret

    def load_training_data(self):
        return Dataset(self.data.x, self.data.y, FLAGS.cs_random_state).shuffle()

    def load_testing_data(self):
        ret = Dataset([], [], FLAGS.cs_random_state)

        for i in range(FLAGS.cs_test_examples):
            ret.x.append(self.eval_data.x[i])
            ret.y.append(self.eval_data.y[i])

        return ret

    def post_process_flags(self):
        FLAGS.cs_train_examples = self.data.get_length()
        FLAGS.cs_test_examples = self.eval_data.get_length()
        FLAGS.cs_total_examples = FLAGS.cs_train_examples + FLAGS.cs_test_examples

    @staticmethod
    def load_ext_data():
        def read_clef_from_file(loc):
            df = pd.read_csv(loc)
            ret_txt, ret_lab = [row['text'] for idx, row in df.iterrows()], [row['label'] for idx, row in df.iterrows()]
            return ret_txt, ret_lab

        data_loc = (FLAGS.cs_prc_data_loc if not FLAGS.cs_use_clef_data else FLAGS.cs_prc_clef_loc)

        if not os.path.isfile(data_loc):
            FLAGS.cs_refresh_data = True

        if FLAGS.cs_refresh_data:
            train_data = (DataLoader.parse_json(FLAGS.cs_raw_data_loc) if not FLAGS.cs_use_clef_data else
                          read_clef_from_file(FLAGS.cs_raw_clef_train_loc))
            dj_eval_data = (DataLoader.parse_json(FLAGS.cs_raw_dj_eval_loc) if not FLAGS.cs_use_clef_data else
                            read_clef_from_file(FLAGS.cs_raw_clef_test_loc))

            if not FLAGS.cs_use_clef_data:
                train_txt = [z[0] for z in train_data]
                eval_txt = [z[0] for z in dj_eval_data]

                train_lab = [z[1] for z in train_data]
                eval_lab = [z[1] for z in dj_eval_data]
            else:
                train_txt, train_lab = train_data
                eval_txt, eval_lab = dj_eval_data

            logging.info('Loading preprocessing dependencies')
            transf.load_dependencies()

            logging.info('Processing train data')
            train_txt, _, train_sent = transf.process_dataset(train_txt)
            logging.info('Processing eval data')
            eval_txt, _, eval_sent = transf.process_dataset(eval_txt)

            train_features, eval_features = DataLoader.process_text_for_transformers(train_txt, eval_txt)

            train_features = DataLoader.convert_data_to_tensorflow_format(train_features)
            eval_features = DataLoader.convert_data_to_tensorflow_format(eval_features)

            train_data = Dataset(list(map(list, zip(train_features.tolist(), train_sent))), train_lab,
                                 random_state=FLAGS.cs_random_state)
            eval_data = Dataset(list(map(list, zip(eval_features.tolist(), eval_sent))), eval_lab,
                                random_state=FLAGS.cs_random_state)

            with open(data_loc, 'wb') as f:
                pickle.dump((train_data, eval_data), f)
            logging.info('Refreshed data, successfully dumped at {}'.format(data_loc))
        else:
            logging.info('Restoring data from {}'.format(data_loc))
            with open(data_loc, 'rb') as f:
                train_data, eval_data = pickle.load(f)

        return train_data, eval_data

    @staticmethod
    def convert_data_to_tensorflow_format(features):
        return DataLoader.pad_seq(features)

    @staticmethod
    def process_text_for_transformers(train_txt, eval_txt):
        if FLAGS.cs_tfm_type == 'bert':
            vocab_file = os.path.join(FLAGS.cs_model_loc, "vocab.txt")
            tokenizer = bert2.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
            train_txt = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in train_txt]
            eval_txt = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in eval_txt]
        else:
            tokenizer = CustomAlbertTokenizer()
            train_txt = tokenizer.tokenize_array(train_txt)
            eval_txt = tokenizer.tokenize_array(eval_txt)

        return train_txt, eval_txt

    @staticmethod
    def parse_json(json_loc):
        with open(json_loc) as f:
            temp_data = json.load(f)

        dl = []
        labels = [0 for _ in range(FLAGS.cs_num_classes)]

        for el in temp_data:
            lab = int(el["label"])
            txt = el["text"]

            labels[lab] += 1
            dl.append([txt, lab])
            print(lab)

        logging.info('{}: {}'.format(json_loc, labels))
        return dl

    @staticmethod
    def pad_seq(inp, ver=0):  # 0 is int, 1 is string
        return pad_sequences(inp, padding="post", maxlen=FLAGS.cs_max_len) if ver == 0 else \
            pad_sequences(inp, padding="post", maxlen=FLAGS.cs_max_len, dtype='str', value='')
