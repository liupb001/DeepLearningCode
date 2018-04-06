from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('/mnist_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.get_variable('weight', shape=[784, 10])
b = tf.get_variable('bias', shape=[10])

y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
loss = -tf.reduce_sum(y * tf.log(y_pred))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

writer = tf.summary.FileWriter('./code_graphs', graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, acc = sess.run([train_op, accuracy], feed_dict={x: batch_x, y: batch_y})

        if i % 50 == 0:
            print('i = {}, acc = {}'.format(i, acc))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

writer.close()
