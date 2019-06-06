import tensorflow as tf
import numpy as np
import math
import time
import os
from utils.data_loader import DataLoader
from models.recurrent import RecurrentModel
from models.embeddings import Embedding
from flags import FLAGS

x = tf.placeholder(tf.int32, (None, FLAGS.max_len), name='x')
x_len = tf.placeholder(tf.int32, (None,), name='x_len')
output_mask = tf.placeholder(tf.bool, (None, FLAGS.max_len), name='output_mask')
y = tf.placeholder(tf.int32, (None, FLAGS.num_classes), name='y')
kp_emb = tf.placeholder(tf.float32, name='kp_emb')
kp_lstm = tf.placeholder(tf.float32, name='kp_lstm')
tf_embed = None


def pad_seq(inp):
    ret = np.full((len(inp), FLAGS.max_len), -1, dtype=np.int32)
    for i in range(len(inp)):
        ret[i][:len(inp[i])] = inp[i]
    return ret


def one_hot(a):
    return np.squeeze(np.eye(FLAGS.num_classes)[np.array(a).reshape(-1)])


def execute_validation(sess, cost, acc, validation_data):
    n_batches = math.ceil(float(FLAGS.validation_examples) / float(FLAGS.batch_size))
    val_loss, val_acc = 0.0, 0.0
    tot_val_ex = 0

    for batch in range(n_batches):
        batch_x, batch_y = get_batch(batch, validation_data, ver='validation')
        tloss, tacc = validation_stats(sess, cost, acc, batch_x, batch_y)
        val_loss += tloss
        val_acc += tacc * len(batch_y)
        tot_val_ex += len(batch_y)

    val_loss /= tot_val_ex
    val_acc /= tot_val_ex
    return 'Val Loss: {:>7.4f} Val Acc: {:>7.4f}% '.format(val_loss, val_acc * 100)


def validation_stats(sess, cost, acc, batch_x, batch_y):
    val_loss = sess.run(
        cost,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )
    val_acc = sess.run(
        acc,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )

    return np.sum(val_loss), val_acc


def batch_stats(sess, batch_x, batch_y, cost, acc):
    train_loss = sess.run(
        cost,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )
    train_acc = sess.run(
        acc,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: 1.0,
            kp_lstm: 1.0
        }
    )

    return np.sum(train_loss), train_acc


def train_neural_network(sess, optimizer, batch_x, batch_y):
    sess.run(
        optimizer,
        feed_dict={
            x: pad_seq(batch_x),
            x_len: [len(el) for el in batch_x],
            output_mask: [[1 if j == len(el) - 1 else 0 for j in range(FLAGS.max_len)] for el in batch_x],
            y: one_hot(batch_y),
            kp_emb: FLAGS.keep_prob_emb,
            kp_lstm: FLAGS.keep_prob_lstm
        }
    )


def get_batch(bid, data, ver='train'):
    batch_x = []
    batch_y = []

    for i in range(FLAGS.batch_size):
        idx = bid * FLAGS.batch_size + i
        if idx >= (FLAGS.train_examples if ver == 'train' else FLAGS.validation_examples):
            break
        batch_x.append(data.x[idx])
        batch_y.append(data.y[idx])

    return batch_x, batch_y


def save_model(sess, epoch):
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(FLAGS.model_dir, 'cb.ckpt'), global_step=epoch)


def main():
    global tf_embed

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tf.logging.info("Loading dataset")
    data_load = DataLoader()

    train_data = data_load.load_training_data()
    validation_data = data_load.load_validation_data()

    tf.logging.info("{} training examples".format(train_data.get_length()))
    tf.logging.info("{} validation examples".format(validation_data.get_length()))

    embed_obj = Embedding()
    tf_embed = tf.placeholder(tf.float32, embed_obj.embed_shape, name='tf_embed')
    embed = embed_obj.construct_embeddings(tf_embed)

    lstm_model = RecurrentModel()
    logits, cost = lstm_model.construct_model(x, x_len, output_mask, y, embed, kp_emb, kp_lstm, adv=FLAGS.adv_train)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

    y_pred = tf.nn.softmax(logits, axis=1, name='y_pred')
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        embed_obj.init_embeddings(sess, tf_embed)

        start = time.time()
        epochs_trav = 0

        tf.logging.info("Starting{}training...".format(' adversarial ' if FLAGS.adv_train else ' '))
        for epoch in range(FLAGS.max_steps):
            epochs_trav += 1
            n_batches = math.ceil(float(FLAGS.train_examples) / float(FLAGS.batch_size))

            n_samples = 0
            epoch_loss = 0.0
            epoch_acc = 0.0

            for i in range(n_batches):
                batch_x, batch_y = get_batch(i, train_data)
                train_neural_network(sess, optimizer, batch_x, batch_y)

                b_loss, b_acc = batch_stats(sess, batch_x, batch_y, cost, acc)
                epoch_loss += b_loss
                epoch_acc += b_acc * len(batch_y)
                n_samples += len(batch_y)

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            if epoch % FLAGS.stat_print_interval == 0:
                log_string = 'Epoch {:>3} Loss: {:>7.4} Acc: {:>7.4f}% '.format(epoch + 1, epoch_loss,
                                                                                epoch_acc * 100)
                if validation_data.get_length() > 0:
                    log_string += execute_validation(sess, cost, acc, validation_data)
                log_string += '({:3.3f} sec/epoch)'.format((time.time() - start) / epochs_trav)

                tf.logging.info(log_string)

                start = time.time()
                epochs_trav = 0

            if epoch % FLAGS.model_save_interval == 0 and epoch != 0:
                save_model(sess, epoch)
                tf.logging.info('Model @ epoch {} saved'.format(epoch + 1))

        tf.logging.info('Training complete. Saving final model...')
        save_model(sess, -1)
        tf.logging.info('Model saved.')

        sess.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
