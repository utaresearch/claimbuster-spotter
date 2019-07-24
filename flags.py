import os
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


# ------------------------- FLAGS FOR 2-CLASS TRAINING -------------------------

# Re-Copy later when needed

# ------------------------- FLAGS FOR 3-CLASS TRAINING -------------------------

# Hardware
flags.DEFINE_list('gpu', [0], 'ID of GPU to use: in range [0, 4]')

# Preprocessing
flags.DEFINE_bool('ner_spacy', False, 'Named entity recognition with spaCy')

# Base directories
flags.DEFINE_string('cb_input_dir', './pt_model', 'Location of pretrained model. Used ONLY when adv/con training')
flags.DEFINE_string('cb_output_dir', './output', 'Location of outputs')
flags.DEFINE_string('cb_data_dir', './data', 'Location of data')
flags.DEFINE_string('tb_dir', './tb_logs', 'Tensorboard location')


# Data
flags.DEFINE_bool('refresh_data', False, 'Re-process ./data/all_data.pickle')
flags.DEFINE_integer('max_len', 200, 'Maximum length of input')
flags.DEFINE_bool('remove_stopwords', False, 'Remove stop words (e.g. the, a, etc.)')
flags.DEFINE_bool('sklearn_oversample', False, 'Oversample underrepresented classes with sklearn')
flags.DEFINE_bool('weight_classes_loss', False, 'Weight classes in CE loss function')
flags.DEFINE_bool('custom_preprc', True, 'Use custom pre-processing')

# Eval
flags.DEFINE_integer('stat_print_interval', 1, 'Numbers of epochs before stats are printed again')
flags.DEFINE_integer('model_save_interval', 1, 'Numbers of epochs before model is saved again')

# Data v2
flags.DEFINE_integer('total_examples', None, 'Total number of examples')
flags.DEFINE_integer('train_examples', None, 'Number of training examples')
flags.DEFINE_integer('test_examples', None, 'Number of testing examples')
flags.DEFINE_integer('random_state', 59, 'State of pseudo-randomness')

# Model architecture
flags.DEFINE_integer('cls_hidden', 0, 'Size of hidden classification layer')

# Optimization
flags.DEFINE_integer('pretrain_steps', 20, 'Number of epochs to run.')
flags.DEFINE_integer('advtrain_steps', 20, 'Number of epochs to run.')
flags.DEFINE_float('lr', 2e-5, 'Learning rate while during optimiation.')

# Regularization
flags.DEFINE_float('l2_reg_coeff', 0.001, 'If val > 0, use L2 Regularization on weights in graph')
flags.DEFINE_float('keep_prob_cls', 0.7, 'Keep probability of classification layer.')

# Word2vec for pre-processing
flags.DEFINE_string('w2v_loc', './data/word2vec/w2v3b_gensim.txt', 'Location of w2v embeddings')
flags.DEFINE_string('w2v_loc_bin', './data/word2vec/w2v3b_gensim.bin', 'Location of w2v embeddings in BINARY form')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_integer('adv_type', 0, '0 for AT, 1 for VAT')
flags.DEFINE_float('adv_coeff', 1.0, 'Coefficient of adversarial loss')
flags.DEFINE_float('perturb_norm_length', 5.0, 'Norm length of adversarial perturbation')

# Output stats
flags.DEFINE_integer('num_classes', 3, 'Number of classes for classification (2 combines NFS and UFS)')

# Transformer
flags.DEFINE_bool('tfm_type', 0, '0 XLNet 1 BERT')
flags.DEFINE_integer('tfm_layers', 12, 'Number of BERT layers.')
flags.DEFINE_bool('tfm_ft_embed', False, 'Train BERT embedding layer')
flags.DEFINE_bool('tfm_ft_pooler', True, 'Train BERT pooler layer')
flags.DEFINE_integer('tfm_ft_enc_layers', 2, 'Last `var` encoding layers are marked as trainable')

# XLNET
flags.DEFINE_string('xlnet_model_loc', './data/xlnet_pretrain', 'Root location of pretrained XLNet files.')
flags.DEFINE_string('xlnet_model_size', 'base', 'Version of XLNet to use: base or large')
flags.DEFINE_bool('use_bfloat16', False, 'Use float16 rather than 32')
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", 0.02, "Initialization std when init is normal.")
flags.DEFINE_float("init_range", 0.1, "Initialization std when init is uniform.")
flags.DEFINE_integer("clamp_len", -1, "Clamp length")

flags.DEFINE_integer("warmup_steps", 5, "number of warmup steps")
flags.DEFINE_float("lr_layer_decay_rate", 1.0, "Top layer: lr[L] = FLAGS.learning_rate. Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", 0.0, "Min lr ratio for cos decay.")
flags.DEFINE_float("clip", 1.0, "Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# BERT
flags.DEFINE_string('bert_model_loc', './data/bert_pretrain', 'Root location of pretrained BERT files.')
flags.DEFINE_string('xlnet_model_size', 'base', 'Version of BERT to use: base or large_wwm')
flags.DEFINE_string('bert_model_hub', 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1', 'Location of BERT on TF hubs.')

# Training
flags.DEFINE_bool('adam', True, 'Adam or RMSProp if False')
flags.DEFINE_bool('restore_and_continue', False, 'Restore previous training session and continue')
flags.DEFINE_integer('batch_size', 12, 'Size of the batch.')

# Locations (must be last due to customization)
flags.DEFINE_string('raw_data_loc', '{}/data_small.json'.format(FLAGS.cb_data_dir), 'Location of raw data')
flags.DEFINE_string('raw_dj_eval_loc', '{}/disjoint_2000.json'.format(FLAGS.cb_data_dir), 'Location of raw data')
flags.DEFINE_string('prc_data_loc', '{}/all_data.pickle'.format(FLAGS.cb_data_dir), 'Location of saved processed data')

FLAGS.xlnet_model_loc = FLAGS.xlnet_model_loc + '_' + FLAGS.xlnet_model_size
FLAGS.bert_model_loc = FLAGS.bert_model_loc + '_' + FLAGS.bert_model_size
if 'large' in any([FLAGS.xlnet_model_size, FLAGS.bert_model_size]):
	FLAGS.tfm_layers = 24


def print_flags():
    tf.logging.info(FLAGS.flag_values_dict())
