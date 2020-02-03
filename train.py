import math
import time
import os
from tqdm import tqdm
from shutil import rmtree
from core.utils.data_loader import DataLoader
from core.utils.flags import FLAGS, print_flags
from absl import logging
import tensorflow as tf
import numpy as np
from core.models.model import ClaimSpotterModel
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from core.utils.compute_ndcg import compute_ndcg

K = tf.keras


def train_model(train_x, train_y, train_len, test_x, test_y, test_len, class_weights, fold):
    assert test_len > 0

    dataset_train = tf.data.Dataset.from_tensor_slices(([x[0] for x in train_x], [x[1] for x in train_x], train_y)).shuffle(
        buffer_size=train_len).batch(FLAGS.cs_batch_size_reg if not FLAGS.cs_adv_train else FLAGS.cs_batch_size_adv)
    dataset_test = tf.data.Dataset.from_tensor_slices(([x[0] for x in test_x], [x[1] for x in test_x], test_y)).shuffle(
        buffer_size=test_len).batch(FLAGS.cs_batch_size_reg if not FLAGS.cs_adv_train else FLAGS.cs_batch_size_adv)

    logging.info("Warming up...")

    model = ClaimSpotterModel(cls_weights=class_weights)
    model.warm_up()

    start_epoch, end_epoch = 0, FLAGS.cs_train_steps

    if FLAGS.cs_restore_and_continue:
        logging.info('Attempting to restore weights from {}'.format(FLAGS.cs_model_dir))

        last_epoch = model.load_custom_model()

        start_epoch += last_epoch + 1
        end_epoch += last_epoch + 1 + FLAGS.cs_train_steps

        logging.info('Restore successful')

    logging.info("Starting{}training...".format(' ' if not FLAGS.cs_adv_train else ' adversarial '))

    aggregated_performance = []
    epochs_trav = 0

    for epoch in range(start_epoch, end_epoch, 1):
        epochs_trav += 1
        epoch_loss, epoch_acc = 0, 0
        start = time.time()

        pbar = tqdm(total=math.ceil(
            train_len / (FLAGS.cs_batch_size_reg if not FLAGS.cs_adv_train else FLAGS.cs_batch_size_adv)))
        for x_id, x_sent, y in dataset_train:
            x = (x_id, x_sent)
            train_batch_loss, train_batch_acc = (model.train_on_batch(x, y) if not FLAGS.cs_adv_train
                                                 else model.adv_train_on_batch(x, y))
            epoch_loss += train_batch_loss
            epoch_acc += train_batch_acc * np.shape(y)[0]
            pbar.update(1)
        pbar.close()

        epoch_loss /= train_len
        epoch_acc /= train_len

        if epoch % FLAGS.cs_stat_print_interval == 0:
            log_string = 'Epoch {:>3} Loss: {:>7.4} Acc: {:>7.4f}% '.format(epoch + 1, epoch_loss, epoch_acc * 100)

            val_loss, val_acc = 0, 0
            val_y, val_pred = [], []

            for x_id, x_sent, y in dataset_test:
                x = (x_id, x_sent)
                val_batch_loss, val_batch_acc = model.stats_on_batch(x, y)
                val_loss += val_batch_loss
                val_acc += val_batch_acc * np.shape(y)[0]

                preds = model.preds_on_batch((x_id, x_sent))
                val_pred = val_pred + preds.numpy().tolist()
                val_y = val_y + y.numpy().tolist()

            val_loss /= test_len
            val_acc /= test_len
            val_pred = np.argmax(val_pred, axis=1)
            f1_mac = f1_score(val_y, val_pred, average='macro')
            f1_wei = f1_score(val_y, val_pred, average='weighted')

            log_string += 'Dev Loss: {:>7.4f} Dev Acc: {:>7.4f} F1-Mac: {:>7.4f} F1-Wei: {:>7.4f}'.format(
                val_loss, val_acc, f1_mac, f1_wei)

            log_string += '({:3.3f} sec/epoch)'.format((time.time() - start) / epochs_trav)

            logging.info(log_string)

            start = time.time()
            epochs_trav = 0

        if epoch % FLAGS.cs_model_save_interval == 0:
            loc = model.save_custom_model(epoch, fold, f1_wei)
            aggregated_performance.append((f1_wei, loc))

    return list(sorted(aggregated_performance, key=lambda x: x[0], reverse=True))[0]


def eval_model(test_x, test_y, test_len, model_loc):
    dataset_test = tf.data.Dataset.from_tensor_slices(
        ([x[0] for x in test_x], [x[1] for x in test_x], test_y)).shuffle(
        buffer_size=test_len).batch(FLAGS.cs_batch_size_reg)

    logging.info("Warming up...")

    model = ClaimSpotterModel()
    model.warm_up()

    logging.info('Attempting to restore weights from {}'.format(model_loc))
    model.load_custom_model(loc=model_loc)
    logging.info('Restore successful')

    logging.info("Starting evaluation...")

    all_y, all_pred = [], []

    pbar = tqdm(total=math.ceil(test_len / FLAGS.cs_batch_size_reg))
    for x_id, x_sent, y in dataset_test:
        preds = model.preds_on_batch((x_id, x_sent))
        all_pred = all_pred + preds.numpy().tolist()
        all_y = all_y + y.numpy().tolist()

        pbar.update(1)
    pbar.close()

    return all_y, all_pred


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(z) for z in FLAGS.cs_gpu])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print_flags()

    if os.path.isdir(FLAGS.cs_tb_dir):
        rmtree(FLAGS.cs_tb_dir)
    if os.path.isdir(FLAGS.cs_model_dir) and not FLAGS.cs_restore_and_continue:
        print('Continue? (y/n) You will overwrite the contents of FLAGS.cs_model_dir ({})'.format(FLAGS.cs_model_dir))
        inp = input().strip('\r\n\t')
        if inp.lower() == 'y':
            rmtree(FLAGS.cs_model_dir)
        else:
            print('Exiting...')
            exit()
    if not os.path.isdir(FLAGS.cs_model_dir) and FLAGS.cs_restore_and_continue:
        raise Exception('Cannot restore from non-existent folder: {}'.format(FLAGS.cs_model_dir))

    logging.info("Loading dataset")
    data_load = DataLoader()

    if FLAGS.cs_k_fold > 1:
        all_data = data_load.load_crossval_data()
        all_data.x = np.array(all_data.x[:10])
        all_data.y = np.array(all_data.y[:10])
        logging.info("{} total cross-validation examples".format(all_data.get_length()))

        kf = KFold(n_splits=FLAGS.cs_k_fold)
        kf.get_n_splits(all_data.x)
        agg_y, agg_pred = [], []

        for iteration, (train_idx, test_idx) in enumerate(kf.split(all_data.x)):
            train_x, test_x = all_data.x[train_idx], all_data.x[test_idx]
            train_y, test_y = all_data.y[train_idx], all_data.y[test_idx]
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

            train_len, val_len, test_len = len(train_y), len(val_y), len(test_y)
            print_str = '|     Running k-fold cross-val iteration #{}: {} train {} val {} test     |'.format(
                iteration + 1, train_len, val_len, test_len)
            horz_str = ''.join(['-' for _ in range(len(print_str))])
            vert_str = '|' + ''.join([' ' for _ in range(len(print_str) - 2)]) + '|'
            logging.info(horz_str); logging.info(vert_str); logging.info(print_str); logging.info(vert_str); logging.info(horz_str);
            logging.info(train_idx)
            logging.info(test_idx)

            res = train_model(train_x, train_y, train_len, val_x, val_y, val_len,
                              DataLoader.compute_class_weights_fold(train_y), iteration)

            cur_y, cur_pred = eval_model(test_x, test_y, test_len, res[1])
            agg_y = np.concatenate((agg_y, cur_y))
            try:
                agg_pred = np.concatenate((agg_pred, cur_pred))
            except Exception as e:
                agg_pred = cur_pred

            print_str = '|     Iteration #{} OK     |'.format(iteration + 1)
            horz_str = ''.join(['-' for _ in range(len(print_str))])
            vert_str = '|' + ''.join([' ' for _ in range(len(print_str) - 2)]) + '|'
            logging.info(horz_str); logging.info(vert_str); logging.info(print_str); logging.info(vert_str); logging.info(horz_str);

        logging.info(agg_y)
        logging.info(agg_pred)

        f1_mac = f1_score(agg_y, np.argmax(agg_pred, axis=1), average='macro')
        f1_wei = f1_score(agg_y, np.argmax(agg_pred, axis=1), average='weighted')
        ndcg = compute_ndcg(agg_y, [x[FLAGS.cs_num_classes - 1] for x in agg_pred])

        target_names = ['NCS', 'CFS']

        logging.info('Final stats | F1-Mac: {:>.4f} F1-Wei: {:>.4f} nDCG: {:>.4f}'.format(
            f1_mac, f1_wei, ndcg))
        print(classification_report(agg_y, np.argmax(agg_pred, axis=1), target_names=target_names, digits=4))
    else:
        train_data = data_load.load_training_data()
        test_data = data_load.load_testing_data()

        logging.info("{} training examples".format(train_data.get_length()))
        logging.info("{} validation examples".format(test_data.get_length()))

        train_model(train_data.x, train_data.y, train_data.get_length(),
                    test_data.x, test_data.y, test_data.get_length(), data_load.class_weights, fold=1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main()
