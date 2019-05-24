import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Data
flags.DEFINE_string('data_dir', './output', 'Location of outputs')
flags.DEFINE_string('vocab_path', './output/vocab.pickle', 'Path to pre-calculated vocab data')
flags.DEFINE_string('data_pkl_loc', './output/prc_data.pickle', 'Location of prc data')
flags.DEFINE_integer('max_len', 150, 'Maximum length of input')

flags.DEFINE_integer('stat_print_interval', 1, 'Numbers of epochs before stats are printed again')
flags.DEFINE_integer('model_save_interval', 25, 'Numbers of epochs before model is saved again')

# Save
flags.DEFINE_string('save_loc', './output', 'Location to save model')

# Data v2
flags.DEFINE_float('train_pct', .70, 'Training percentage')
flags.DEFINE_float('validation_pct', .05, 'Validation percentage')
flags.DEFINE_float('test_pct', .25, 'Testing percentage')
flags.DEFINE_integer('total_examples', None, 'Total number of examples')
flags.DEFINE_integer('train_examples', None, 'Number of training examples')
flags.DEFINE_integer('validation_examples', None, 'Number of validation examples')
flags.DEFINE_integer('test_examples', None, 'Number of testing examples')

flags.DEFINE_integer('random_state', 42, 'State of consistent pseudo-randomness')

# Embeddings
flags.DEFINE_string('w2v_loc', 'data/word2vec/GoogleNews-vectors-negative300.bin', 'Location of w2v embeddings')
flags.DEFINE_bool('transfer_learn_w2v', False, 'Train on top of w2v embeddings')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_float('perturb_norm_length', 5.0, 'Norm length of adversarial perturbation')

# Parameters for building the graph
flags.DEFINE_string('adv_training_method', None, 'How adversarial training is to be undertaken'
flags.DEFINE_float('adv_reg_coeff', 1.0, 'Regularization coefficient of adversarial loss')

# Output stats
flags.DEFINE_integer('num_classes', 3, 'Number of classes for classification')

# Training
flags.DEFINE_integer('batch_size', 256, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')

# Model architecture
flags.DEFINE_bool('bidir_lstm', False, 'Whether to build a bidirectional LSTM.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 512, 'Number of hidden units in the LSTM.')

# Vocabulary and embeddings
flags.DEFINE_integer('embedding_dims', 300, 'Dimensions of embedded vector.')
flags.DEFINE_bool('normalize_embeddings', False, 'Normalize word embeddings by vocab frequency')

# Optimization
flags.DEFINE_integer('max_steps', 1000, 'Number of epochs to run.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate while fine-tuning.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0, 'Learning rate decay factor')

# Regularization
flags.DEFINE_float('max_grad_norm', 1.0, 'Clip the global gradient norm to this value.')
flags.DEFINE_float('keep_prob_lstm', 0.5, 'Keep probability LSTM network.')
flags.DEFINE_float('keep_prob_emb', 0.5, 'Keep probability on embedding layer.')
