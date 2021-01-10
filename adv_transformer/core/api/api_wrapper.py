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

import os
import numpy as np
from adv_transformer.core.models.model import ClaimSpotterModel
from adv_transformer.core.utils.data_loader import DataLoader
from adv_transformer.core.utils import transformations as transf
from adv_transformer.core.utils.flags import FLAGS
from absl import logging
import tensorflow as tf

from transformers import AutoTokenizer


class ClaimSpotterAPI:
    def __init__(self):
        logging.set_verbosity(logging.INFO)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.cs_gpu])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        self.return_strings = ['Non-factual sentence', 'Check-worthy factual statement']
        # self.tokenizer = AdvFullTokenizer(os.path.join(FLAGS.cs_model_loc, "vocab.txt"), do_lower_case=True)
        self.tokenizer = AutoTokenizer.from_pretrained(FLAGS.cs_tfm_type)

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
            (self._create_tfm_features(sentence_features[0]), sentence_features[1])).batch(FLAGS.cs_batch_size_reg)

    def _retrieve_model_preds(self, dataset):
        ret = []
        for x, x_sent in dataset:
            ret = ret + self.model.preds_on_batch((x, x_sent)).numpy().tolist()
        return self._apply_activation(ret)

    def _create_tfm_features(self, sentence_list):
        features = self.tokenizer(sentence_list)['input_ids']
        return DataLoader.pad_seq(features)

    @staticmethod
    def _apply_activation(x):
        r = FLAGS.cs_ca_r

        if not FLAGS.cs_custom_activation:
            inter = np.apply_along_axis(np.exp, 1, x)
        else:
            inter = np.apply_along_axis(lambda z: np.exp(r * z) / (np.exp(r * z) + 1), 1, x)

        return np.apply_along_axis(lambda z: z / z.sum(), 1, inter)

    @staticmethod
    def _extract_info(sentence_list):
        r_sentence_list = [transf.transform_sentence_complete(x) for x in sentence_list]
        r_sentiment_list = [transf.get_sentiment(x) for x in sentence_list]
        return r_sentence_list, r_sentiment_list
