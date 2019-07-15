import tensorflow as tf
import sys
import json
import os
from .adv_losses import apply_adversarial_perturbation
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
    def build_bert_transformer_raw(x_id, x_mask, x_segment, adv):
        tf.logging.info('Building BERT transformer')

        config = BertConfig.from_json_file(os.path.join(FLAGS.bert_model_loc, 'bert_config.json'))
        model = BertModel(config=config, is_training=True, input_ids=x_id, input_mask=x_mask, token_type_ids=x_segment)

        bert_outputs = model.get_pooled_output()

        init_checkpoint = os.path.join(FLAGS.bert_model_loc, 'bert_model.ckpt')

        assignment_map, initialized_variable_names = \
            get_assignment_map_from_checkpoint(tf.trainable_variables(), init_checkpoint)

        print(assignment_map)
        print(initialized_variable_names)

        restore_op = tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        return ([None, bert_outputs], restore_op) if not adv else (bert_outputs, restore_op)
