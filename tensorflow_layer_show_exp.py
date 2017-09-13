import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# add layer to create a model
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

hide_layer = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(hide_layer, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.1)
# optimizer = tf.train.AdadeltaOptimizer(0.1) choose
# optimizer = tf.train.RMSPropOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
plt.ion()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5001):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 100 == 0:
            plt.cla()
            plt.scatter(x_data, y_data, c='cyan', label='test data')
            prediction_y, loss_y = sess.run([prediction, loss], feed_dict={xs: x_data, ys: y_data})
            lines = plt.plot(x_data, prediction_y, 'r-', lw=5, label='learning lines')
            text = plt.text(-0.35, -0.25, 'loss=%.5f' % loss_y, fontdict={'size': 15})
            plt.legend(loc='upper left')
            plt.ylim(-0.5, 1.4)
            plt.draw()
            plt.pause(0.2)

            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
plt.show()
