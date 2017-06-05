import tensorflow as tf
import numpy as np

input_a = np.array([[1, 1, 2], [2, 3, 4]], dtype=np.float32)
input_b = np.array([[True, False], [True, True]])
input_c = np.array([[1.3, 1.2, 2.3], [2., 3., 2.3]], dtype=np.float32)

input_a_sum_column = tf.reduce_sum(input_a, reduction_indices=0)
input_a_sum_row = tf.reduce_sum(input_a, reduction_indices=1, keep_dims=True)

input_a_prod_column = tf.reduce_prod(input_a, reduction_indices=0)
input_a_prod_row = tf.reduce_prod(input_a, reduction_indices=1, keep_dims=True)

input_a_min = tf.reduce_min(input_a, reduction_indices=1)
input_a_max = tf.reduce_max(input_a, reduction_indices=1)
input_a_mean = tf.reduce_mean(input_a, reduction_indices=1, keep_dims=True)

input_b_and = tf.reduce_all(input_b, reduction_indices=1)
input_b_or = tf.reduce_any(input_b, reduction_indices=1)

input_accum = tf.accumulate_n(inputs=[input_a, input_c])
input_cum = tf.cumsum(x=[input_a_sum_column, input_a_prod_column])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(input_a_sum_column), '\n', sess.run(input_a_sum_row))
    print(sess.run(input_a_prod_column), '\n', sess.run(input_a_prod_row))
    print(sess.run(input_a_min), '\n', sess.run(input_a_max), '\n', sess.run(input_a_mean))
    print(sess.run(input_accum), '\n', sess.run(input_cum))
