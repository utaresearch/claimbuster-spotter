import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Data
flags.DEFINE_string('data_pkl_loc', './output/prc_data.pickle', 'Train on top of w2v embeddings')
flags.DEFINE_float('train_pct', .70, 'Training percentage')
flags.DEFINE_float('validation_pct', .05, 'Validation percentage')
flags.DEFINE_float('test_pct', .25, 'Testing percentage')
flags.DEFINE_integer('total_examples', None, 'Total number of examples')
flags.DEFINE_integer('train_examples', None, 'Number of training examples')
flags.DEFINE_integer('validation_examples', None, 'Number of validation examples')
flags.DEFINE_integer('test_examples', None, 'Number of testing examples')

flags.DEFINE_integer('random_state', 42, 'State of consistent pseudo-randomness')

# Embeddings
flags.DEFINE_bool('transfer_learn_w2v', False, 'Train on top of w2v embeddings')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_float('perturb_norm_length', 5.0,
                   'Norm length of adversarial perturbation to be '
                   'optimized with validation. '
                   '5.0 is optimal on IMDB with virtual adversarial training. ')

# Virtual adversarial training parameters
flags.DEFINE_integer('num_power_iteration', 1, 'The number of power iteration')
flags.DEFINE_float('small_constant_for_finite_diff', 1e-1,
                   'Small constant for finite difference method')

# Parameters for building the graph
flags.DEFINE_string('adv_training_method', None,
                    'The flag which specifies training method. '
                    '""    : non-adversarial training (e.g. for running the '
                    '        semi-supervised sequence learning model) '
                    '"rp"  : random perturbation training '
                    '"at"  : adversarial training '
                    '"vat" : virtual adversarial training '
                    '"atvat" : at + vat ')
flags.DEFINE_float('adv_reg_coeff', 1.0,
                   'Regularization coefficient of adversarial loss.')

# Output stats
flags.DEFINE_integer('num_classes', 3, 'Number of classes for classification')

flags.DEFINE_string('data_dir', '/tmp/IMDB',
                    'Directory path to preprocessed text dataset.')
flags.DEFINE_string('vocab_path', None,
                    'Path to pre-calculated vocab frequency data. If '
                    'None, use FLAGS.data_dir/vocab_freq.txt.')
flags.DEFINE_string('vocab_freq_path', None,
                    'Path to pre-calculated vocab frequency data. If '
                    'None, use FLAGS.data_dir/vocab_freq.txt.')

# Training
flags.DEFINE_integer('batch_size', 64, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')

# Model architecture
flags.DEFINE_bool('bidir_lstm', True, 'Whether to build a bidirectional LSTM.')
# flags.DEFINE_bool('dropout', True, 'Whether to build a bidirectional LSTM.')
flags.DEFINE_bool('single_label', True, 'Whether the sequence has a single '
                  'label, for optimization.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 512,
                     'Number of hidden units in the LSTM.')
flags.DEFINE_integer('cl_num_layers', 1,
                     'Number of hidden layers of classification model.')
flags.DEFINE_integer('cl_hidden_size', 30,
                     'Number of hidden units in classification layer.')
flags.DEFINE_integer('num_candidate_samples', -1,
                     'Num samples used in the sampled output layer.')
flags.DEFINE_bool('use_seq2seq_autoencoder', False,
                  'If True, seq2seq auto-encoder is used to pretrain. '
                  'If False, standard language model is used.')

# Vocabulary and embeddings
flags.DEFINE_integer('embedding_dims', 256, 'Dimensions of embedded vector.')
flags.DEFINE_bool('normalize_embeddings', True,
                  'Normalize word embeddings by vocab frequency')

# Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate while fine-tuning.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0,
                   'Learning rate decay factor')

# Regularization
flags.DEFINE_float('max_grad_norm', 1.0,
                   'Clip the global gradient norm to this value.')
flags.DEFINE_float('keep_prob_emb', 1.0, 'keep probability on embedding layer. '
                   '0.5 is optimal on IMDB with virtual adversarial training.')
flags.DEFINE_float('keep_prob_lstm_out', 1.0,
                   'keep probability on lstm output.')
flags.DEFINE_float('keep_prob_cl_hidden', 1.0,
                   'keep probability on classification hidden layer')