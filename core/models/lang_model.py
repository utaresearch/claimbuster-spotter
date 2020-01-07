from . import bert2
from ..utils.flags import FLAGS


class LanguageModel:
    def __init__(self):
        pass

    @staticmethod
    def build_transformer():  # independent of tfm type
        bert_params = bert2.params_from_pretrained_ckpt(FLAGS.cs_model_loc)
        bert_params.hidden_dropout = 1 - FLAGS.cs_kp_tfm_hidden
        bert_params.attention_dropout = 1 - FLAGS.cs_kp_tfm_atten
        return bert2.BertModelLayer.from_params(bert_params, name=FLAGS.cs_tfm_type)

    # @staticmethod
    # def test_albert():
    #     model_name = "albert_base"
    #     model_dir = bert2.fetch_tfhub_albert_model(model_name, ".models")
    #     model_params = bert2.albert_params(model_name)
    #     return bert2.BertModelLayer.from_params(model_params, name="albert")
    #
    #     # use in Keras Model here, and call model.build()
    #
    #     bert2.load_albert_weights(l_bert, albert_dir)  # should be called after model.build()