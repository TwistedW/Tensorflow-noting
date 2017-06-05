import tensorflow as tf
import numpy as np

input_a = np.array([1, 0, 2, 5])
input_b = np.array([2, 3, 4, 5])
input_c = np.array([[True, False], [True, True]])
input_d = np.array([1, 1, 2, 2, 3, 5, 6])
input_e = np.array([0, 2, 3, 1, 4])

a_argmin = tf.argmin(input_a)
a_argmax = tf.argmax(input_a)
a_listdiff = tf.setdiff1d(input_a, input_b)
c_where = tf.where(input_c)
d_unique = tf.unique(input_d)
e_invert_permutation = tf.invert_permutation(input_e)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(a_argmin), '\n', sess.run(a_argmax), '\n', sess.run(a_listdiff), '\n', sess.run(c_where))
    print(sess.run(d_unique), '\n', sess.run(e_invert_permutation))