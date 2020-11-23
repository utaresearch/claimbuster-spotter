# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

import pandas as pd
import pickle
import os
import json
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from absl import logging
from . import transformations as transf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .flags import FLAGS

from transformers import AutoTokenizer


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
        self.data, self.eval_data = (self.load_ext_data() if FLAGS.cs_k_fold <= 1 else self.load_kfold_data())

        self.class_weights = self.compute_class_weights()
        logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()

    def compute_class_weights(self):
        return compute_class_weight('balanced', classes=[z for z in range(FLAGS.cs_num_classes)], y=self.data.y)

    @staticmethod
    def compute_class_weights_fold(y_list):
        return compute_class_weight('balanced', classes=[z for z in range(FLAGS.cs_num_classes)], y=y_list)

    def load_training_data(self):
        ret = Dataset(self.data.x, self.data.y, FLAGS.cs_random_state)
        ret.shuffle()
        return ret

    def load_testing_data(self):
        ret = Dataset(self.eval_data.x, self.eval_data.y, FLAGS.cs_random_state)
        ret.shuffle()
        return ret

    def load_crossval_data(self):
        ret = Dataset(self.data.x, self.data.y, FLAGS.cs_random_state)
        ret.shuffle()
        return ret

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
    def load_kfold_data():
        data_loc = FLAGS.cs_prc_data_loc

        if not os.path.isfile(data_loc):
            FLAGS.cs_refresh_data = True

        if FLAGS.cs_refresh_data:
            train_data = DataLoader.parse_json(FLAGS.cs_raw_kfold_data_loc)

            train_txt = [z[0] for z in train_data]
            train_lab = [z[1] for z in train_data]

            logging.info('Loading preprocessing dependencies')
            transf.load_dependencies()

            logging.info('Processing data')
            train_txt, _, train_sent = transf.process_dataset(train_txt)

            train_features, _ = DataLoader.process_text_for_transformers(train_txt, [])
            train_features = DataLoader.convert_data_to_tensorflow_format(train_features)

            train_data = Dataset(list(map(list, zip(train_features.tolist(), train_sent))), train_lab,
                                 random_state=FLAGS.cs_random_state)

            with open(data_loc, 'wb') as f:
                pickle.dump(train_data, f)
            logging.info('Refreshed data, successfully dumped at {}'.format(data_loc))
        else:
            logging.info('Restoring data from {}'.format(data_loc))
            with open(data_loc, 'rb') as f:
                train_data = pickle.load(f)

        return train_data, None

    @staticmethod
    def convert_data_to_tensorflow_format(features):
        return DataLoader.pad_seq(features)

    @staticmethod
    def process_text_for_transformers(train_txt, eval_txt):
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.cs_tfm_type)
        train_txt = tokenizer(train_txt)['input_ids'] if len(train_txt) > 0 else []
        eval_txt = tokenizer(eval_txt)['input_ids'] if len(eval_txt) > 0 else []
        return train_txt, eval_txt

    @staticmethod
    def parse_json(json_loc):
        with open(json_loc, encoding=FLAGS.cs_data_file_encoding) as f:
            temp_data = json.load(f)

        three_class_file = ('deprecated' in json_loc)

        dl = []
        labels = [0 for _ in range(FLAGS.cs_num_classes)]

        for el in temp_data:
            lab = int(el["label"]) + (1 if three_class_file else 0) - (1 if (int(el["label"]) != -1 and three_class_file) else 0)
            txt = el["text"]

            labels[lab] += 1
            dl.append([txt, lab])

        logging.info('{}: {}'.format(json_loc, labels))
        return dl

    @staticmethod
    def pad_seq(inp, ver=0):  # 0 is int, 1 is string
        return pad_sequences(inp, padding="post", maxlen=FLAGS.cs_max_len) if ver == 0 else \
            pad_sequences(inp, padding="post", maxlen=FLAGS.cs_max_len, dtype='str', value='')
