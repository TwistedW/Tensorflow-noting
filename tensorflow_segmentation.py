import tensorflow as tf
import numpy as np

input_a = np.array([[1, 1, 2], [2, 3, 4], [3, 1, 1], [2, 4, 6]])
a_seg_sum = tf.segment_sum(data=input_a, segment_ids=[0, 1, 1, 1])
a_seg_prod = tf.segment_prod(data=input_a, segment_ids=[0, 0, 1, 1])
a_seg_max = tf.segment_max(data=input_a, segment_ids=[0, 0, 0, 1])
a_seg_min = tf.segment_min(data=input_a, segment_ids=[1, 1, 1, 1])
a_seg_mean = tf.segment_mean(data=input_a, segment_ids=[0, 0, 0, 1])
a_seg_sum_num = tf.unsorted_segment_sum(data=input_a, segment_ids=[0, 1, 1, 0], num_segments=2)
a_sparse_seg_sum = tf.sparse_segment_sum(data=input_a, indices=[0, 1, 2], segment_ids=[0, 0, 1])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(a_seg_sum), '\n', sess.run(a_seg_prod), '\n', sess.run(a_seg_max), '\n', sess.run(a_seg_min))
    print(sess.run(a_seg_mean), '\n', sess.run(a_seg_sum_num), '\n', sess.run(a_sparse_seg_sum))