import tensorflow as tf
import numpy as np

a = np.array([1,2,3])
a2 = np.array([[1,2],[3,4],[2,2]])
a3 = np.array([[1,1,1],[1,2,3],[0,3,4]], dtype=np.float32)
b = tf.diag(a)
a1 = tf.diag_part(b)
b1 = tf.trace(b)
b2 = tf.matrix_determinant(tf.to_float(b))#must be type of float
c = tf.matmul(b, a2)
d = tf.matrix_inverse(input=a3, adjoint=False)
d1 = tf.cholesky(tf.to_float(b))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(b), '\n', sess.run(a1), '\n', sess.run(b1), '\n', sess.run(b2), '\n', sess.run(c))
    print(sess.run(d),'\n', sess.run(d1))
