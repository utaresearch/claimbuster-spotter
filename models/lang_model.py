import tensorflow as tf
import sys
import json
import os
from .adv_losses import get_adversarial_perturbation
from .bert_model import BertConfig, BertModel, get_assignment_map_from_checkpoint

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
        assert not adv

        tf.logging.info('Building BERT transformer')

        import tensorflow_hub as hub

        bert_module = hub.Module(FLAGS.bert_model_hub, trainable=FLAGS.bert_trainable)
        bert_inputs = dict(
            input_ids=x_id,
            input_mask=x_mask,
            segment_ids=x_segment)
        bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)

        return [None, bert_outputs["pooled_output"]] if not adv else bert_outputs

    @staticmethod
    def build_bert_transformer_raw(x_id, x_mask, x_segment, adv=False, orig_embed=None, reg_loss=None):
        tf.logging.info('Building{}BERT transformer'.format(' adversarial ' if adv else ' '))

        hparams = LanguageModel.load_bert_pretrain_hyperparams()

        config = BertConfig(vocab_size=hparams['vocab_size'], hidden_size=hparams['hidden_size'],
                            num_hidden_layers=hparams['num_hidden_layers'],
                            num_attention_heads=hparams['num_attention_heads'],
                            intermediate_size=hparams['intermediate_size'], hidden_act=hparams['hidden_act'],
                            hidden_dropout_prob=hparams['hidden_dropout_prob'],
                            attention_probs_dropout_prob=hparams['attention_probs_dropout_prob'],
                            max_position_embeddings=hparams['max_position_embeddings'],
                            type_vocab_size=hparams['type_vocab_size'],
                            initializer_range=hparams['initializer_range'])

        perturb = get_adversarial_perturbation(orig_embed, reg_loss)

        model = BertModel(config=config, is_training=True, input_ids=x_id, input_mask=x_mask, token_type_ids=x_segment,
                          adv=adv, perturb=perturb)
        bert_outputs = model.get_pooled_output()

        init_checkpoint = os.path.join(FLAGS.bert_model_loc, 'bert_model.ckpt')
        assignment_map, _ = \
            get_assignment_map_from_checkpoint(tf.trainable_variables(), init_checkpoint)

        restore_op = tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        return ([model.get_embedding_output(), bert_outputs], restore_op) if not adv else (bert_outputs, restore_op)

    @staticmethod
    def load_bert_pretrain_hyperparams():
        with open(os.path.join(FLAGS.bert_model_loc, 'bert_config.json'), 'r') as f:
            data = json.load(f)

        return data
