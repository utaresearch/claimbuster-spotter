import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Preprocessing
flags.DEFINE_bool('noun_rep', False, 'Replace nouns with identifiers')
flags.DEFINE_bool('full_tags', False, 'Replace all words with tags')
flags.DEFINE_bool('ner_spacy', True, 'Named entity recognition with spaCy')

# Base directories
flags.DEFINE_string('output_dir', './output', 'Location of outputs')
# flags.DEFINE_string('output_dir', './data/disjoint_2000/', 'Location of outputs')
flags.DEFINE_string('data_dir', './data', 'Location of data')

# Data
flags.DEFINE_integer('max_len', 200, 'Maximum length of input')
flags.DEFINE_bool('undersample_NFS', False, 'Reduce NFS to level of CFS')
flags.DEFINE_bool('smote_synthetic', False, '[NOT WORKING] Oversample imabalanced classes using imblearn')
flags.DEFINE_bool('sklearn_oversample', True, 'Oversample underrepresented classes with sklearn')

# Eval
flags.DEFINE_bool('disjoint_data', True, 'Use custom non-train set data to evaluate model')
flags.DEFINE_string('custom_prc_data_loc', './data/disjoint_2000/prc_data.pickle', 'Location of custom data file')
flags.DEFINE_string('custom_vocab_loc', './data/disjoint_2000/vocab.pickle', 'Location of custom vocab file')


flags.DEFINE_integer('stat_print_interval', 1, 'Numbers of epochs before stats are printed again')
flags.DEFINE_integer('model_save_interval', 25, 'Numbers of epochs before model is saved again')

# Data v2
flags.DEFINE_float('train_pct', 0.95, 'Training percentage')
flags.DEFINE_float('validation_pct', 0.05, 'Validation percentage')
flags.DEFINE_float('test_pct', 0.00, 'Testing percentage')
flags.DEFINE_integer('total_examples', None, 'Total number of examples')
flags.DEFINE_integer('train_examples', None, 'Number of training examples')
flags.DEFINE_integer('validation_examples', None, 'Number of validation examples')
flags.DEFINE_integer('test_examples', None, 'Number of testing examples')
flags.DEFINE_integer('random_state', 59, 'State of pseudo-randomness')

# Model architecture
flags.DEFINE_bool('bidir_lstm', False, 'Whether to build a bidirectional LSTM.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 64, 'Number of hidden units in the LSTM.')

# Optimization
flags.DEFINE_integer('max_steps', 1000, 'Number of epochs to run.')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate while during optimiation.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0, 'Learning rate decay factor')

# Regularization
flags.DEFINE_float('l2_reg_coeff', 0.005, 'If val > 0, use L2 Regularization on weights in graph')
flags.DEFINE_float('max_grad_norm', 1.0, 'Clip the global gradient norm to this value.')
flags.DEFINE_float('keep_prob_lstm', 0.7, 'Keep probability LSTM network.')
flags.DEFINE_float('keep_prob_emb', 0.7, 'Keep probability on embedding layer.')

# Embeddings
flags.DEFINE_string('w2v_loc', 'data/word2vec/GoogleNews-vectors-negative300.bin', 'Location of w2v embeddings')
flags.DEFINE_bool('train_embed', True, 'Train on top of w2v embeddings')
flags.DEFINE_integer('embedding_dims', 300, 'Dimensions of embedded vector.')
flags.DEFINE_bool('normalize_embeddings', False, 'Normalize word embeddings by vocab frequency')
flags.DEFINE_bool('random_init_oov', True, 'Use np.random.normal init for unknown embeddings. 0-fill if False')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_float('perturb_norm_length', 5.0, 'Norm length of adversarial perturbation')

# Parameters for building the graph
flags.DEFINE_string('adv_training_method', None, 'How adversarial training is to be undertaken')
flags.DEFINE_float('adv_reg_coeff', 1.0, 'Regularization coefficient of adversarial loss')

# Output stats
flags.DEFINE_integer('num_classes', 3, 'Number of classes for classification')

# Training
flags.DEFINE_integer('batch_size', 256, 'Size of the batch.')

# Locations (must be last due to customization)
flags.DEFINE_string('model_dir', FLAGS.output_dir, 'Location of model save')
flags.DEFINE_string('vocab_loc', '{}/vocab.pickle'.format(FLAGS.output_dir), 'Path to pre-calculated vocab data')
flags.DEFINE_string('prc_data_loc', '{}/prc_data.pickle'.format(FLAGS.output_dir), 'Location of processed data')
flags.DEFINE_string('raw_data_loc', '{}/data_small.json'.format(FLAGS.data_dir), 'Location of raw data')
# flags.DEFINE_string('raw_data_loc', './data/disjoint_2000.json', 'Location of raw data')
