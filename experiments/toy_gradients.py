import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 3), name='y')
    my_dense = tf.layers.Dense(3, name="dense_layer")
    eps = tf.constant(.5)

    yhat = my_dense(x)
    
    yhat_norm = tf.nn.softmax(yhat)
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(yhat_norm, axis=1))
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(y, yhat)

    grad = tf.stop_gradient(tf.gradients(cost, x))
    yhat_adv = my_dense(x + grad / tf.norm(grad) * eps)
    yhat_adv_norm = tf.nn.softmax(yhat_adv)
    correct_adv = tf.equal(tf.argmax(y, axis=1), tf.argmax(yhat_adv_norm, axis=1))
    cost_adv = tf.nn.softmax_cross_entropy_with_logits_v2(y, yhat_adv)

    optimize = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost + cost_adv)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100000):
            sess.run(optimize, feed_dict={
                x: [[1], [2], [3]],
                y: [[1,0,0],[0,1,0],[0,0,1]]
            })
            if i % 1000 == 0:
                print(i + 1,
                    np.sum(sess.run(cost, feed_dict={x: [[1], [2], [3]], y: [[1,0,0],[0,1,0],[0,0,1]]})),
                    np.mean(sess.run(correct, feed_dict={x: [[1], [2], [3]], y: [[1,0,0],[0,1,0],[0,0,1]]})),
                    np.mean(sess.run(cost_adv, feed_dict={x: [[1], [2], [3]], y: [[1,0,0],[0,1,0],[0,0,1]]})),
                    np.mean(sess.run(correct_adv, feed_dict={x: [[1], [2], [3]], y: [[1,0,0],[0,1,0],[0,0,1]]}))
                )
        print()

        print(sess.run(yhat_norm, feed_dict={
            x: [[1]]
        }))
        print(sess.run(yhat_norm, feed_dict={
            x: [[2]]
        }))
        print(sess.run(yhat_norm, feed_dict={
            x: [[3]]
        }))
        print(sess.run(yhat_adv_norm, feed_dict={
            x: [[3]],
            y: [[0,0,1]]
        }))
        print(sess.run(grad, feed_dict={
            x: [[3]],
            y: [[0,0,1]]
        }))