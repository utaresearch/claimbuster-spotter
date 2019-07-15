import tensorflow as tf

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 3), name='y')
    my_dense = tf.layers.Dense(3, name="dense_layer")
    eps = tf.constant(2.)

    yhat = my_dense(x)
    
    yhat_norm = tf.nn.softmax(yhat)
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(y, yhat)
    optimize = tf.train.AdamOptimizer().minimize(cost)

    grad = tf.gradients(cost, x)
    yhat_adv = my_dense(x + grad / tf.norm(grad) * eps)
    yhat_adv_norm = tf.nn.softmax(yhat_adv)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('epoch ')
        for i in range(10000):
            if i % 1000 == 0:
                print(i + 1, end=' ', flush=True)

            sess.run(optimize, feed_dict={
                x: [[1], [2], [3]],
                y: [[1,0,0],[0,1,0],[0,0,1]]
            })
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