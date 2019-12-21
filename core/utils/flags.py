from absl import app, flags, logging
import sys

FLAGS = flags.FLAGS


# ------------------------- FLAGS FOR 2-CLASS TRAINING -------------------------

# Re-Copy later when needed

# ------------------------- FLAGS FOR 3-CLASS TRAINING -------------------------

# Hardware
flags.DEFINE_list('cs_gpu', [0], 'ID of GPU to use: in range [0, 4]')

# Preprocessing
flags.DEFINE_bool('cs_ner_spacy', False, 'Named entity recognition with spaCy')

# Base directories
flags.DEFINE_string('cs_model_dir', './output', 'Location of model (both input and output)')
flags.DEFINE_string('cs_model_ckpt', 'bert_claimspotter.ckpt', 'Filename of model .ckpt')
flags.DEFINE_string('cs_data_dir', './data', 'Location of data')
flags.DEFINE_string('cs_tb_dir', './tb_logs', 'Tensorboard location')

flags.DEFINE_bool('cs_adv_train', False, 'Use adversarial training?')

# Data
flags.DEFINE_bool('cs_use_clef_data', False, 'Use CLEF data, rather than ClaimBuster')
flags.DEFINE_bool('cs_combine_ours_clef_data', False, 'Combine CLEF with ClaimBuster data')
flags.DEFINE_bool('cs_refresh_data', False, 'Re-process ./data/all_data.pickle')
flags.DEFINE_integer('cs_max_len', 200, 'Maximum length of input')
flags.DEFINE_bool('cs_remove_stopwords', False, 'Remove stop words (e.g. the, a, etc.)')
flags.DEFINE_bool('cs_sklearn_oversample', False, 'Oversample underrepresented classes with sklearn')
flags.DEFINE_bool('cs_weight_classes_loss', False, 'Weight classes in CE loss function')
flags.DEFINE_bool('cs_custom_preprc', True, 'Use custom pre-processing')

# Eval
flags.DEFINE_integer('cs_stat_print_interval', 1, 'Numbers of epochs before stats are printed again')
flags.DEFINE_integer('cs_model_save_interval', 1, 'Numbers of epochs before model is saved again')

# Data v2
flags.DEFINE_integer('cs_total_examples', None, 'Total number of examples')
flags.DEFINE_integer('cs_train_examples', None, 'Number of training examples')
flags.DEFINE_integer('cs_test_examples', None, 'Number of testing examples')
flags.DEFINE_integer('cs_random_state', 59, 'State of pseudo-randomness')

# Model architecture
flags.DEFINE_integer('cs_cls_hidden', 0, 'Size of hidden classification layer')
flags.DEFINE_float('cs_kp_cls', 0.7, 'Keep probability of dropout in FC')

# Optimization
flags.DEFINE_integer('cs_train_steps', 100, 'Number of epochs to run.')
flags.DEFINE_float('cs_lr', 5e-5, 'Learning rate while during optimiation.')

# Regularization
flags.DEFINE_float('cs_l2_reg_coeff', 5e-4, 'If val > 0, use L2 Regularization on weights in graph')

# Adversarial and virtual adversarial training parameters.
flags.DEFINE_integer('cs_perturb_id', 0, "Index in [('pos', 'seg', 'tok'), ('pos', 'seg'), ('pos', 'tok'), ('seg', 'tok'), ('pos',), ('seg',), ('tok',)] to perturb")
flags.DEFINE_integer('cs_adv_type', 0, '0 for AT, 1 for VAT')
flags.DEFINE_float('cs_adv_coeff', 0.5, 'Coefficient of adversarial loss')
flags.DEFINE_bool('cs_combine_reg_adv_loss', True, 'Add loss of regular and adversarial loss during training')
flags.DEFINE_float('cs_perturb_norm_length', 3.0, 'Norm length of adversarial perturbation')

# Output stats
flags.DEFINE_integer('cs_num_classes', 3, 'Number of classes for classification (2 combines NFS and UFS)')
flags.DEFINE_bool('cs_alt_two_class_combo', False, 'Combine CFS and UFS instead when num_classes=2.')

# Transformer
flags.DEFINE_bool('cs_tfm_type', 1, '0 XLNet 1 BERT')
flags.DEFINE_integer('cs_tfm_layers', 12, 'Number of BERT layers.')
flags.DEFINE_bool('cs_tfm_ft_embed', False, 'Train BERT embedding layer')
flags.DEFINE_bool('cs_tfm_ft_pooler', True, 'Train BERT pooler layer')
flags.DEFINE_integer('cs_tfm_ft_enc_layers', 2, 'Last n encoding layers are marked as trainable')
flags.DEFINE_float('cs_kp_tfm_atten', 0.8, 'Keep probability of attention dropout in Transformer')
flags.DEFINE_float('cs_kp_tfm_hidden', 0.8, 'Keep probability of hidden dropout in Transformer')

# BERT
flags.DEFINE_string('cs_bert_model_size', 'base', 'Version of BERT to use: base or large_wwm')

# Training
flags.DEFINE_bool('cs_adam', True, 'Adam or RMSProp if False')
flags.DEFINE_bool('cs_restore_and_continue', False, 'Restore previous training session and continue')
flags.DEFINE_integer('cs_batch_size_reg', 24, 'Size of the batch.')
flags.DEFINE_integer('cs_batch_size_adv', 12, 'Size of the batch when adversarial training.')


def clean_argv(inp):
	ret = [inp[0]]
	del inp[0]

	for x in inp:
		x_name = x.split('=')[0]
		if 'cs_' in x_name:
			ret.append(x_name)

	return ret


FLAGS(clean_argv(sys.argv))

# Locations (must be last due to customization)
flags.DEFINE_string('cs_bert_model_loc', '{}/bert_pretrain'.format(FLAGS.cs_data_dir), 'Root location of pretrained BERT files.')
flags.DEFINE_string('cs_raw_data_loc', '{}/data_small.json'.format(FLAGS.cs_data_dir), 'Location of raw data')
flags.DEFINE_string('cs_raw_dj_eval_loc', '{}/disjoint_2000.json'.format(FLAGS.cs_data_dir), 'Location of raw data')
flags.DEFINE_string('cs_raw_clef_train_loc', '{}/CT19-T1-Training.csv'.format(FLAGS.cs_data_dir), 'Location of raw CLEF .csv data')
flags.DEFINE_string('cs_raw_clef_test_loc', '{}/CT19-T1-Test.csv'.format(FLAGS.cs_data_dir), 'Location of raw CLEF .csv data')
flags.DEFINE_string('cs_prc_data_loc', '{}/all_data.pickle'.format(FLAGS.cs_data_dir), 'Location of saved processed data')
flags.DEFINE_string('cs_prc_clef_loc', '{}/all_clef_data.pickle'.format(FLAGS.cs_data_dir), 'Location of saved processed CLEF data')

FLAGS.cs_bert_model_loc = FLAGS.cs_bert_model_loc + '_' + FLAGS.cs_bert_model_size
if any(['large' in FLAGS.cs_bert_model_size]):
	FLAGS.cs_tfm_layers *= 2
	FLAGS.cs_tfm_ft_enc_layers *= 2
	FLAGS.cs_batch_size_reg //= 3
	FLAGS.cs_batch_size_adv //= 3


def print_flags():
	logging.info(FLAGS.flag_values_dict())
