import numpy as np
import os
from ..models.model import ClaimSpotterModel
from ..models import bert2
from ..utils.data_loader import DataLoader
from ..utils import transformations as transf
from ..utils.flags import FLAGS
from absl import logging
import tensorflow as tf
import numpy as np


class ClaimSpotterAPI:
    def __init__(self):
        logging.set_verbosity(logging.INFO)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.cs_gpu])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        self.return_strings = ['Non-factual sentence', 'Check-worthy factual statement']
        self.tokenizer = bert2.bert_tokenization.FullTokenizer(os.path.join(FLAGS.cs_model_loc, "vocab.txt"),
                                                               do_lower_case=True)

        transf.load_dependencies()

        self.model = ClaimSpotterModel()
        self.model.warm_up()
        self.model.load_custom_model()

    def subscribe_cmdline_query(self):
        print('Enter a sentence to process')
        return self._retrieve_model_preds(self._prc_sentence_list([input().strip('\n\r\t ')]))

    def single_sentence_query(self, sentence):
        return self._retrieve_model_preds(self._prc_sentence_list([sentence.strip('\n\r\t ')]))

    def batch_sentence_query(self, sentence_list):
        sentence_list = [x.strip('\n\r\t ') for x in sentence_list]
        return self._retrieve_model_preds(self._prc_sentence_list(sentence_list))

    def _prc_sentence_list(self, sentence_list):
        sentence_features = self._extract_info(sentence_list)
        return tf.data.Dataset.from_tensor_slices(
            (self._create_bert_features(sentence_features[0]), sentence_features[1])).batch(FLAGS.cs_batch_size_reg)

    def _retrieve_model_preds(self, dataset):
        ret = []
        for x, x_sent in dataset:
            val = self.model.preds_on_batch((x, x_sent)).numpy()
            print(val)
            val = (val - val.min()) / (val.max() - val.min())
            print(val)
            val = val / val.sum()
            print(val)
            ret = ret + val.tolist()
        return ret

    def _create_bert_features(self, sentence_list):
        features = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
                    for x in sentence_list]
        return DataLoader.pad_seq(features)

    @staticmethod
    def _extract_info(sentence_list):
        r_sentence_list = [transf.transform_sentence_complete(x) for x in sentence_list]
        r_sentiment_list = [transf.get_sentiment(x) for x in sentence_list]
        return r_sentence_list, r_sentiment_list
