from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../mnist_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

W1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_image, W1))
pool1 = max_pool(conv1)

W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, W2))
pool2 = max_pool(conv2)

pool2_fc = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])

W_fc = weight_variable([7 * 7 * 64, 1024])
b_fc = bias_variable([1024])
fc = tf.matmul(pool2_fc, W_fc) + b_fc

keep_prob = tf.placeholder(tf.float32)
dropout = tf.nn.dropout(fc, keep_prob)

W_out = weight_variable([1024, 10])
b_out = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(fc, W_out) + b_out)

loss = -tf.reduce_sum(y * tf.log(y_pred))

train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)


correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

writer = tf.summary.FileWriter('./code_graphs', graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, acc = sess.run([train_op, accuracy], feed_dict={x: batch_x, y: batch_y,
                                                           keep_prob: 0.6})

        if i % 20 == 0:
            print('i={}, acc={}'.format(i, acc))

    print('test acc: ', sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                      y: mnist.test.labels,
                                                      keep_prob: 1}))

writer.close()
