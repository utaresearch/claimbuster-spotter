import numpy as np
import os
from model import ClaimBusterModel
from models import bert2
from utils.data_loader import DataLoader
from utils import transformations as transf
from flags import FLAGS
from absl import logging
import tensorflow as tf


class ClaimBusterAPI:
    def __init__(self):
        logging.set_verbosity(logging.ERROR)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        self.return_strings = ['Non-factual statement', 'Unimportant factual statement', 'Salient factual statement']
        self.tokenizer = bert2.bert_tokenization.FullTokenizer(os.path.join(FLAGS.bert_model_loc, "vocab.txt"),
                                                               do_lower_case=True)

        transf.load_dependencies()

        self.model = ClaimBusterModel()
        self.model.warm_up()
        self.model.load_custom_model()

    def prc_sentence(self, sentence):
        sentence, sent = self.extract_info(sentence)
        ds = tf.data.Dataset.from_tensor_slices([self.create_bert_features(sentence)]).batch(1)

        return next(iter(ds)).numpy(), sent

    def create_bert_features(self, sentence):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))

    def subscribe_cmdline_query(self):
        print('Enter a sentence to process')
        sentence_duple = self.prc_sentence(input().strip('\n\r\t '))
        print(sentence_duple)

        return self.model.preds_on_batch(sentence_duple[0])

    def direct_sentence_query(self, sentence):
        sentence_duple = self.prc_sentence(sentence.strip('\n\r\t '))
        return self.model.preds_on_batch(sentence_duple[0])

    @staticmethod
    def extract_info(sentence):
        sentence = transf.transform_sentence_complete(sentence)
        sent = transf.get_sentiment(sentence)

        return sentence, sent
