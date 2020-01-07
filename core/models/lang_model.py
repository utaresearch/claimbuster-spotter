from . import bert2
from ..utils.flags import FLAGS
import tensorflow_hub as hub
import tensorflow as tf

K = tf.keras


class LanguageModel:
    def __init__(self):
        pass

    @staticmethod
    def build_transformer():  # independent of tfm type
        if FLAGS.cs_tfm_type == 'albert':
            return LanguageModel.test_albert()

        bert_params = bert2.params_from_pretrained_ckpt(FLAGS.cs_model_loc)
        bert_params.hidden_dropout = 1 - FLAGS.cs_kp_tfm_hidden
        bert_params.attention_dropout = 1 - FLAGS.cs_kp_tfm_atten
        return bert2.BertModelLayer.from_params(bert_params, name=FLAGS.cs_tfm_type)

    @staticmethod
    def test_albert():
        albert_module = hub.Module(
            "https://tfhub.dev/google/albert_base/2",
            trainable=True)
        albert_inputs = dict(
            input_ids=K.layers.Input(shape=(FLAGS.cs_max_len,), dtype='int32')
        )
        albert_outputs = albert_module(albert_inputs, signature="tokens", as_dict=True)
        return albert_outputs["pooled_output"]