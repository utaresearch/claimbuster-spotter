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
import tensorflow as tf
from bert import params_from_pretrained_ckpt
from adv_transformer.core.models.advbert.model import AdvBertModelLayer
from adv_transformer.core.utils.flags import FLAGS


class LanguageModel:
    def __init__(self):
        pass

    @staticmethod
    def build_transformer():
        bert_params = params_from_pretrained_ckpt(FLAGS.cs_model_loc)
        bert_params.hidden_dropout = 1 - FLAGS.cs_kp_tfm_hidden
        bert_params.attention_dropout = 1 - FLAGS.cs_kp_tfm_atten

        return AdvBertModelLayer.from_params(bert_params, name=FLAGS.cs_tfm_type)

    """
    @staticmethod
    def test_albert():
        albert_module = hub.Module(
            "https://tfhub.dev/google/albert_base/2",
            trainable=True)
        albert_inputs = dict(
            input_ids=tf.keras.layers.Input(shape=(FLAGS.cs_max_len,), dtype='int32')
        )
        albert_outputs = albert_module(albert_inputs, signature="tokens", as_dict=True)
        return albert_outputs["pooled_output"]
    """