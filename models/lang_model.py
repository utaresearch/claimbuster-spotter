import tensorflow as tf
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import sys
from .adv_losses import apply_adversarial_perturbation
sys.path.append('..')
from flags import FLAGS


class LanguageModel:
    def __init__(self):
        pass

    @staticmethod
    def build_bert_transformer(x_id, x_mask, x_segment, adv):
        tf.logging.info('Building BERT transformer')

        import tensorflow_hub as hub

        bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=False)
        bert_inputs = dict(
            input_ids=x_id,
            input_mask=x_mask,
            segment_ids=x_segment)
        bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)

        return None, bert_outputs["pooled_output"] if not adv else bert_outputs

    @staticmethod
    def build_embed_lstm(x, x_len, output_mask, embed, kp_lstm, orig_embed, reg_loss, adv):
        if adv:
            x_embed = apply_adversarial_perturbation(orig_embed, reg_loss)
        else:
            x = tf.unstack(x, axis=1)
            for i in range(len(x)):
                x[i] = tf.nn.embedding_lookup(embed, x[i])
            x = tf.stack(x, axis=1)
            x_embed = tf.identity(x, name='x_embed')

        x = x_embed

        if not FLAGS.bidir_lstm:
            tf.logging.info('Building uni-directional LSTM')
            output, _ = LanguageModel.build_unidir_lstm_component(x, x_len, kp_lstm, adv)
        else:
            tf.logging.info('Building bi-directional LSTM')
            output, _ = LanguageModel.build_bidir_lstm_component(x, x_len, kp_lstm, adv)

        if FLAGS.bidir_lstm:
            output_fw, output_bw = output
            output_fw = tf.boolean_mask(output_fw, output_mask)
            output_bw = output_bw[:, 0, :]
            output = tf.concat([output_fw, output_bw], axis=1)
        else:
            output = output[:, -1, :]

        return (x_embed, output) if not adv else output

    @staticmethod
    def build_lstm(x, x_len, output_mask, kp_lstm, adv):
        x = tf.cast(x, tf.float32)

        if not FLAGS.bidir_lstm:
            tf.logging.info('Building uni-directional LSTM')
            output, _ = LanguageModel.build_unidir_lstm_component(x, x_len, kp_lstm, adv)
        else:
            tf.logging.info('Building bi-directional LSTM')
            output, _ = LanguageModel.build_bidir_lstm_component(x, x_len, kp_lstm, adv)

        if FLAGS.bidir_lstm:
            output_fw, output_bw = output
            output_fw = tf.boolean_mask(output_fw, output_mask)
            output_bw = output_bw[:, 0, :]
            output = tf.concat([output_fw, output_bw], axis=1)
        else:
            output = output[:, -1, :]

        return output

    @staticmethod
    def build_unidir_lstm_component(x, x_len, kp_lstm, adv):
        lstm = tf.nn.rnn_cell.MultiRNNCell([LanguageModel.get_lstm(cell_num, kp_lstm, adv, direc=0)
                                            for cell_num in range(FLAGS.rnn_num_layers)])
        return tf.nn.dynamic_rnn(cell=lstm, sequence_length=x_len, inputs=x, dtype=tf.float32)

    @staticmethod
    def build_bidir_lstm_component(x, x_len, kp_lstm, adv):
        assert FLAGS.rnn_num_layers == 1

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([LanguageModel.get_lstm(cell_num, kp_lstm, adv, direc=0)
                                               for cell_num in range(FLAGS.rnn_num_layers)])
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([LanguageModel.get_lstm(cell_num, kp_lstm, adv, direc=1)
                                               for cell_num in range(FLAGS.rnn_num_layers)])

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x, sequence_length=x_len, dtype=tf.float32)

    @staticmethod
    def get_lstm(cell_id, kp_lstm, adv, direc):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(FLAGS.rnn_cell_size, reuse=adv,
                                           name='lstm_cell_{}_{}'.format(cell_id, ('fwd' if direc == 0 else 'bwd')))
        return tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=kp_lstm)
