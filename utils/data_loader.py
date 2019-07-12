import tensorflow as tf
import tensorflow_hub as hub
from bert import run_classifier
from bert import tokenization
from keras.preprocessing.text import Tokenizer
from sklearn.utils import resample
import numpy as np
import pandas as pd
import pickle
import os
import shutil
import json
import sys
from . import transformations as transf
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

cwd = os.getcwd()
root_dir = None

for root, dirs, files in os.walk(cwd):
    for file in files:
        if file.endswith("ac_bert.txt"):
            root_dir = root

if cwd != root_dir:
    from ..flags import FLAGS
else:
    sys.path.append('..')
    from flags import FLAGS

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
        xlen, ylen = len(self.x), len(self.y)
        if xlen != ylen:
            raise ValueError("size of x != size of y ({} != {})".format(xlen, ylen))
        return xlen


class DataLoader:
    def __init__(self, train_data=None, val_data=None, test_data=None):
        assert FLAGS.num_classes == 2 or FLAGS.num_classes == 3

        self.data, self.eval_data, self.vocab = self.load_ext_data(train_data, val_data, test_data)

        if FLAGS.num_classes == 2:
            self.convert_3_to_2()

        self.class_weights = self.compute_class_weights()
        tf.logging.info('Class weights computed to be {}'.format(self.class_weights))

        self.data.shuffle()
        self.post_process_flags()

    def convert_3_to_2(self):
        self.data.y = [(1 if self.data.y[i] == 2 else 0) for i in range(len(self.data.y))]
        self.eval_data.y = [(1 if self.eval_data.y[i] == 2 else 0) for i in range(len(self.eval_data.y))]

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
    def load_ext_data(train_data_in, val_data_in, test_data_in):
        train_data, test_data, vocab = None, None, None

        # if train_data_in is not None and val_data_in is not None and test_data_in is not None:
        #     FLAGS.refresh_data = True

        if FLAGS.refresh_data:
            train_data = DataLoader.parse_json(FLAGS.raw_data_loc) if train_data_in is None else train_data_in
            dj_eval_data = DataLoader.parse_json(FLAGS.raw_dj_eval_loc) if test_data_in is None else test_data_in

            # train_data = resample(train_data, n_samples=10, random_state=FLAGS.random_state)
            # dj_eval_data = resample(dj_eval_data, n_samples=10, random_state=FLAGS.random_state)

            train_txt = [z[0] for z in train_data]
            eval_txt = [z[0] for z in dj_eval_data]

            train_lab = [z[1] + 1 for z in train_data]
            eval_lab = [z[1] + 1 for z in dj_eval_data]

            tf.logging.info('Loading preprocessing dependencies')
            transf.load_dependencies()

            tf.logging.info('Processing train data')
            train_txt, train_pos, train_sent = transf.process_dataset(train_txt)
            tf.logging.info('Processing eval data')
            eval_txt, eval_pos, eval_sent = transf.process_dataset(eval_txt)

            if not FLAGS.bert_model:
                tokenizer = Tokenizer()

                tokenizer.fit_on_texts(np.concatenate((train_txt, eval_txt)))
                train_seq = tokenizer.texts_to_sequences(train_txt)
                eval_seq = tokenizer.texts_to_sequences(eval_txt)

                train_data = Dataset(list(zip(train_seq, train_pos, train_sent)), train_lab, random_state=FLAGS.random_state)
                eval_data = Dataset(list(zip(eval_seq, eval_pos, eval_sent)), eval_lab, random_state=FLAGS.random_state)
                vocab = tokenizer.word_index
            else:
                train_df = pd.DataFrame(data=zip(train_txt, train_lab), columns=['x', 'y'])
                eval_df = pd.DataFrame(data=zip(eval_txt, eval_lab), columns=['x', 'y'])

                train_df = train_df.apply(lambda x: run_classifier.InputExample(guid=None, text_a=x['x'], text_b=None,
                                                                                label=x['y']), axis=1)

                eval_df = eval_df.apply(lambda x: run_classifier.InputExample(guid=None, text_a=x['x'], text_b=None,
                                                                              label=x['y']), axis=1)

                tokenizer = DataLoader.create_tokenizer_from_hub_module()

                train_features = run_classifier.convert_examples_to_features(train_df, [z for z in range(FLAGS.num_classes)],
                                                                             FLAGS.max_len, tokenizer)
                eval_features = run_classifier.convert_examples_to_features(eval_df, [z for z in range(FLAGS.num_classes)],
                                                                            FLAGS.max_len, tokenizer)

                train_data = Dataset(list(zip(train_features, train_pos, train_sent)), train_lab, random_state=FLAGS.random_state)
                eval_data = Dataset(list(zip(eval_features, eval_pos, eval_sent)), eval_lab, random_state=FLAGS.random_state)

            with open(FLAGS.prc_data_loc, 'wb') as f:
                pickle.dump((train_data, eval_data, vocab), f)
            tf.logging.info('Refreshed data, successfully dumped at {}'.format(FLAGS.prc_data_loc))
            print("FOOOOOOOOOOOOOOOOOOO " + os.getcwd())

            if os.path.exists(FLAGS.cb_output_dir):
                shutil.rmtree(FLAGS.cb_output_dir)
            os.mkdir(FLAGS.cb_output_dir)
        else:
            tf.logging.info('Restoring data from {}'.format(FLAGS.prc_data_loc))
            with open(FLAGS.prc_data_loc, 'rb') as f:
                train_data, eval_data, vocab = pickle.load(f)

        if not FLAGS.bert_model:
            assert vocab is not None

        return train_data, eval_data, vocab

    @staticmethod
    def create_tokenizer_from_hub_module():
        os.environ["TFHUB_CACHE_DIR"] = '/home/kevin/bert_cache'

        bert_module = hub.Module(FLAGS.bert_model_hub, trainable=False)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

        return tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

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
