import math
import time
from tqdm import tqdm
import os
from utils.data_loader import DataLoader
from model import ClaimBusterModel
from flags import FLAGS, print_flags
from absl import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, classification_report
from utils.compute_ndcg import compute_ndcg
from model import ClaimBusterModel

K = tf.keras


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.gpu])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print_flags()

    if not os.path.isdir(FLAGS.cb_model_dir):
        raise Exception('Cannot restore from non-existent folder: {}'.format(FLAGS.cb_model_dir))

    logging.info("Loading dataset")
    data_load = DataLoader()

    test_data = data_load.load_testing_data()
    logging.info("{} testing examples".format(test_data.get_length()))

    dataset_test = tf.data.Dataset.from_tensor_slices(([x[0] for x in test_data.x], test_data.y)).shuffle(
        buffer_size=test_data.get_length()).batch(FLAGS.batch_size_reg)

    logging.info("Warming up...")

    model = ClaimBusterModel(cls_weights=data_load.class_weights)
    model.warm_up()

    logging.info('Attempting to restore weights from {}'.format(FLAGS.cb_model_dir))
    model.load_custom_model()
    logging.info('Restore successful')

    logging.info("Starting evaluation...")

    eval_loss, eval_acc = 0, 0
    all_y, all_pred = [], []

    pbar = tqdm(total=math.ceil(len(test_data.y) / FLAGS.batch_size_reg))
    for x_id, y in dataset_test:
        eval_batch_loss, eval_batch_acc = model.stats_on_batch(x_id, y)
        eval_loss += eval_batch_loss
        eval_acc += eval_batch_acc * np.shape(y)[0]

        preds = model.preds_on_batch(x_id)
        all_pred = all_pred + list(preds.numpy())
        all_y = all_y + list(y.numpy())

        pbar.update(1)
    pbar.close()

    eval_loss /= test_data.get_length()
    eval_acc /= test_data.get_length()

    all_pred_argmax = np.argmax(all_pred, axis=1)

    f1score = f1_score(all_y, all_pred_argmax, average='weighted')
    ndcg = compute_ndcg(all_y, [x[FLAGS.num_classes - 1] for x in all_pred])

    target_names = (['NFS', 'UFS', 'CFS'] if FLAGS.num_classes == 3 else ['NFS/UFS', 'CFS'])

    print('Final stats | Loss: {:>7.4} Acc: {:>7.4f}% F1: {:>.4f} nDCG: {:>.4f}'.format(
        eval_loss, eval_acc * 100, f1score, ndcg))
    print(classification_report(all_y, all_pred_argmax, target_names=target_names, digits=4))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main()
