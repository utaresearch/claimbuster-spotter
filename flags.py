import tensorflow as tf
import os

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
flags.DEFINE_string('cb_output_dir', './output', 'Location of outputs')
flags.DEFINE_string('cb_data_dir', './data', 'Location of data')

# Data
flags.DEFINE_bool('refresh_data', False, 'Re-process ./data/all_data.pickle')
flags.DEFINE_integer('max_len', 200, 'Maximum length of input')
flags.DEFINE_bool('remove_stopwords', False, 'Remove stop words (e.g. the, a, etc.)')
flags.DEFINE_bool('sklearn_oversample', False, 'Oversample underrepresented classes with sklearn')
flags.DEFINE_bool('weight_classes_loss', False, 'Weight classes in CE loss function')
flags.DEFINE_list('addition_vocab', ['./data/disjoint_2000/vocab.pickle'], 'Additional corpuses to sample vocab data from')
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
flags.DEFINE_bool('use_bert_hub', True, 'Use TF Hub for BERT rather than raw construction')
flags.DEFINE_string('bert_model_hub', 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1', 'Location of BERT on TF hubs.')
flags.DEFINE_string('bert_model_loc', './data/bert_pretrain', 'Root location of pretrained BERT files.')
flags.DEFINE_integer('bert_layers', 12, 'Number of BERT layers.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 16, 'Number of hidden units in the LSTM.')
flags.DEFINE_integer('cls_hidden', 0, 'Size of hidden classification layer')

# Optimization
flags.DEFINE_integer('max_steps', 10, 'Number of epochs to run.')
# flags.DEFINE_float('learning_rate', 0.001, 'Learning rate while during optimiation.')

# Regularization
flags.DEFINE_float('l2_reg_coeff', 0.001, 'If val > 0, use L2 Regularization on weights in graph')
flags.DEFINE_float('keep_prob_cls', 0.7, 'Keep probability of classification layer.')

# Embeddings
flags.DEFINE_integer('embed_type', 1, '0 for word2vec, 1 for Stanford glove')
flags.DEFINE_string('w2v_loc', './data/word2vec/w2v3b_gensim.txt', 'Location of w2v embeddings')
flags.DEFINE_string('w2v_loc_bin', './data/word2vec/w2v3b_gensim.bin', 'Location of w2v embeddings in BINARY form')
flags.DEFINE_bool('train_embed', False, 'Train on top of w2v embeddings')  # we don't have enough data to train embed
flags.DEFINE_integer('embedding_dims', 100, 'Dimensions of embedded vector.')
flags.DEFINE_bool('random_init_oov', False, 'Use np.random.normal init for unknown embeddings. 0-fill if False')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_bool('adv_train', False, 'Train using adversarial perturbations')
flags.DEFINE_float('adv_coeff', 1.0, 'Coefficient of adversarial loss')
flags.DEFINE_float('perturb_norm_length', 6.0, 'Norm length of adversarial perturbation')

# Output stats
flags.DEFINE_integer('num_classes', 3, 'Number of classes for classification (2 combines NFS and UFS)')

# Training
flags.DEFINE_bool('bert_trainable', True, 'Train BERT weights')
flags.DEFINE_bool('bert_ft_embed', False, 'Train BERT embedding layer')
flags.DEFINE_integer('bert_ft_enc_layers', 2, 'Last `var` encoding layers are marked as trainable')
flags.DEFINE_bool('adam', True, 'Adam or RMSProp if False')
flags.DEFINE_bool('restore_and_continue', False, 'Restore previous training session and continue')
flags.DEFINE_integer('batch_size', 16, 'Size of the batch.')

# Locations (must be last due to customization)
flags.DEFINE_string('raw_data_loc', '{}/data_small.json'.format(FLAGS.cb_data_dir), 'Location of raw data')
flags.DEFINE_string('raw_dj_eval_loc', '{}/disjoint_2000.json'.format(FLAGS.cb_data_dir), 'Location of raw data')
flags.DEFINE_string('prc_data_loc', '{}/all_data.pickle'.format(FLAGS.cb_data_dir), 'Location of saved processed data')

if not os.path.isfile(FLAGS.prc_data_loc):
    FLAGS.refresh_data = True


def print_flags():
    tf.logging.info(FLAGS.flag_values_dict())
