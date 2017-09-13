"""
Through tensorflow to creat a liner data
show the Weights and bias to test and verify it.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Generate data
x_date = np.random.rand(400).astype(np.float32)
noise = np.random.normal(0, 0.05, x_date.shape)
y_date = x_date*2 + 0.5 + noise#used for verification

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -3, 3))
biases = tf.Variable(tf.zeros([1])+0.01)
#the initialization of the value is used for the next learning change

y = Weights*x_date + biases
#maching learning to construct a pile of data

loss = tf.reduce_mean(tf.square(y - y_date))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
#flexible training optimization error

init = tf.initialize_all_variables()
### create tensorflow structure start ###

sess = tf.Session()
sess.run(init) #activate neural network
plt.ion()
for step in range(1001):
    sess.run(train)
    if step % 20 == 0:
        plt.cla()
        plt.scatter(x_date, y_date, label='test data')
        print(step, sess.run(Weights), sess.run(biases))
        prediction_y = sess.run(y)
        lines = plt.plot(x_date, prediction_y, 'r-', lw=5, label='learning lines')
        plt.text(0.35, 0.5, 'loss=%.5f' % sess.run(loss), fontdict={'size': 15, 'color': 'green'})
        plt.legend(loc='upper left')
        plt.pause(0.1)
plt.show()