import tensorflow as tf
import sys
import json
import os
from .adv_losses import apply_adversarial_perturbation
from .bert import BertConfig, BertModel

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


class LanguageModel:
    def __init__(self):
        pass

    @staticmethod
    def build_bert_transformer_hub(x_id, x_mask, x_segment, adv):
        tf.logging.info('Building BERT transformer')

        import tensorflow_hub as hub

        bert_module = hub.Module(FLAGS.bert_model_hub, trainable=FLAGS.bert_trainable)
        bert_inputs = dict(
            input_ids=x_id,
            input_mask=x_mask,
            segment_ids=x_segment)
        bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)

        return None, bert_outputs["pooled_output"] if not adv else bert_outputs

    @staticmethod
    def build_bert_transformer_raw(x_id, x_mask, x_segment, adv):
        tf.logging.info('Building BERT transformer')

        config = BertConfig(vocab_size=32000, hidden_size=512,
                            num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

        model = BertModel(config=config, is_training=True,
                          input_ids=x_id, input_mask=x_mask, token_type_ids=x_segment)

        bert_outputs = model.get_pooled_output()

        return None, bert_outputs["pooled_output"] if not adv else bert_outputs

    @staticmethod
    def load_bert_pretrain_hyperparams():
        data = json.load(os.path.join(FLAGS.bert_model_loc, ''))