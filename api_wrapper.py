from bert import run_classifier
import numpy as np
import os
from model import ClaimBusterModel
from utils.data_loader import DataLoader
from utils import transformations as transf
from flags import FLAGS
import tensorflow as tf


class ClaimBusterAPI:
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'

        self.return_strings = ['Non-factual statement', 'Unimportant factual statement', 'Salient factual statement']
        self.tokenizer = DataLoader.create_tokenizer_from_hub_module()

        data_load = DataLoader()
        self.vocab = data_load.vocab

        transf.load_dependencies()

        graph = tf.get_default_graph()
        self.cb_model = ClaimBusterModel(data_load.vocab, data_load.class_weights, restore=True, adv=False)
        self.cb_model.load_model(graph, train=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(tf.global_variables_initializer())
        self.cb_model.load_model(self.sess, graph)

    def prc_sentence(self, sentence, vocab):
        sentence, pos, sent = self.extract_info(sentence, vocab)

        input_examples = [run_classifier.InputExample(guid="", text_a=sentence, text_b=None, label=0)]
        input_features = run_classifier.convert_examples_to_features(input_examples,
                                                                     [z for z in range(FLAGS.num_classes)],
                                                                     FLAGS.max_len, self.tokenizer)

        return input_features, pos, sent

    def subscribe_cmdline_query(self):
        print('Enter a sentence to process')
        sentence_tuple = self.prc_sentence(input().strip('\n\r\t '), self.vocab)

        return self.cb_model.get_preds(self.sess, sentence_tuple)[0]

    def direct_sentence_query(self, sentence):
        sentence_tuple = self.prc_sentence(sentence.strip('\n\r\t '), self.vocab)
        return self.cb_model.get_preds(self.sess, sentence_tuple)[0]

    @staticmethod
    def extract_info(sentence, vocab):
        sentence = transf.transform_sentence_complete(sentence)

        sent = transf.get_sentiment(sentence)
        pos = transf.process_sentence_full_tags(sentence)

        return sentence, pos, sent
