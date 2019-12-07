from .bert2 import BertModelLayer
import os
import json

import sys
sys.path.append('..')
from flags import FLAGS


class LanguageModel:
    def __init__(self):
        pass

    @staticmethod
    def build_bert():
        hparams = LanguageModel.load_bert_pretrain_hyperparams()

        bert_model = BertModelLayer(**BertModelLayer.Params(
            vocab_size=hparams['vocab_size'],  # embedding params
            use_token_type=True,
            use_position_embeddings=True,
            token_type_vocab_size=2,

            num_layers=hparams['num_hidden_layers'],  # transformer encoder params
            hidden_size=hparams['hidden_size'],
            hidden_dropout=FLAGS.kp_tfm_hidden,
            intermediate_size=hparams['intermediate_size'],
            intermediate_activation=hparams['hidden_act'],

            adapter_size=None,  # see arXiv:1902.00751 (adapter-BERT)

            shared_layer=False,  # True for ALBERT (arXiv:1909.11942)
            embedding_size=None,  # None for BERT, wordpiece embedding size for ALBERT

            name="bert"  # any other Keras layer params
        ))

        return bert_model

    @staticmethod
    def load_bert_pretrain_hyperparams():
        with open(os.path.join(FLAGS.bert_model_loc, 'bert_config.json'), 'r') as f:
            data = json.load(f)

        return data