import sys
import json
import os
from .adv_losses import get_adversarial_perturbation
from .bert_model import BertConfig, BertModel, get_assignment_map_from_checkpoint
from .xlnet.xlnet import XLNetConfig, XLNetModel, create_run_config
from .xlnet.model_utils import init_from_checkpoint
import tensorflow as tf

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
    def build_xlnet_transformer_raw(x_id, x_mask, x_segment, kp_tfm_atten, kp_tfm_hidden,
                                    adv=False, orig_embed=None, reg_loss=None, restore=False):
        assert not adv  # for now

        x_mask = tf.cast(x_mask, tf.float32)

        xlnet_config = XLNetConfig(json_path=os.path.join(FLAGS.xlnet_model_loc, 'xlnet_config.json'))
        run_config = create_run_config(is_training=True, is_finetune=True, FLAGS=FLAGS, dropout=kp_tfm_hidden,
                                       dropatt=kp_tfm_atten)

        xlnet_model = XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=x_id,
            seg_ids=x_segment,
            input_mask=x_mask)

        summary = xlnet_model.get_pooled_out(summary_type="last")

        if not restore:
            tf.logging.info('Retrieving pre-trained XLNET weights')
            init_from_checkpoint(os.path.join(FLAGS.xlnet_model_loc, 'model', 'xlnet_model.ckpt'))
            tf.logging.info('Successfully retrieved XLNET weights')
        else:
            tf.logging.info('Will wait to retrieve complete weights from cb.ckpt')

        print(summary)

        return summary


    @staticmethod
    def build_bert_transformer_raw(x_id, x_mask, x_segment, kp_tfm_atten, kp_tfm_hidden,
                                   adv=False, orig_embed=None, reg_loss=None, restore=False):
        tf.logging.info('Building{}BERT transformer'.format(' adversarial ' if adv else ' '))

        hparams = LanguageModel.load_bert_pretrain_hyperparams()

        config = BertConfig(vocab_size=hparams['vocab_size'], hidden_size=hparams['hidden_size'],
                            num_hidden_layers=hparams['num_hidden_layers'],
                            num_attention_heads=hparams['num_attention_heads'],
                            intermediate_size=hparams['intermediate_size'], hidden_act=hparams['hidden_act'],
                            hidden_dropout_prob=1-kp_tfm_hidden,
                            attention_probs_dropout_prob=1-kp_tfm_atten,
                            max_position_embeddings=hparams['max_position_embeddings'],
                            type_vocab_size=hparams['type_vocab_size'],
                            initializer_range=hparams['initializer_range'])

        perturb = get_adversarial_perturbation(orig_embed, reg_loss) if adv else None

        model = BertModel(config=config, is_training=True, input_ids=x_id, input_mask=x_mask, token_type_ids=x_segment,
                          adv=adv, perturb=perturb)
        bert_outputs = model.get_pooled_output()

        if not restore:
            tf.logging.info('Retrieving pre-trained BERT weights')

            init_checkpoint = os.path.join(FLAGS.bert_model_loc, 'bert_model.ckpt')
            assignment_map, _ = \
                get_assignment_map_from_checkpoint(tf.trainable_variables(), init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        else:
            tf.logging.info('Will wait to retrieve complete weights from cb.ckpt')

        print(bert_outputs)
        exit()

        return (model.get_embedding_output(), bert_outputs) if not adv else bert_outputs

    @staticmethod
    def load_bert_pretrain_hyperparams():
        with open(os.path.join(FLAGS.bert_model_loc, 'bert_config.json'), 'r') as f:
            data = json.load(f)

        return data
