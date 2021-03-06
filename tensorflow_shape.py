import tensorflow as tf
import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9])
b = tf.reshape(a, [3,-1])
c = np.array([[[1,2,3],[4,5,6]],[[6,7,8],[9,10,11]]])
d = tf.reshape(c, [-1,3])
e = tf.expand_dims(d, -1)
f = tf.expand_dims(d, 0)
g = tf.slice(c, [0,0,0], [2,1,3])
h, i = tf.split(axis=0, num_or_size_splits=2, value=c)
t1 = np.array([[1,2,3],[4,5,6]])
t2 = np.array([[7,8,9],[10,11,12]])
t3 = np.array([[1,1,1],[2,2,2]])
j = tf.concat(values=[t1,t2], axis=0)
k = tf.concat(values=[t1,t2], axis=1)
l = tf.stack(values=[t1,t2,t3], axis=0)
m = tf.reverse(tensor=c, axis=[1,0])
n = tf.transpose(c, [1,0,2])
o = tf.one_hot([0,2,1,3],5,3.,1.,-1,tf.float32)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('c shape:',c.shape)
    print(sess.run(b), '\n', sess.run(d).shape, '\n', sess.run(e).shape, '\n', sess.run(f).shape)
    print(sess.run(g))
    print(sess.run(h),'\n', sess.run(i).shape)
    print(sess.run(j),'\n',sess.run(k))
    print(sess.run(l))
    print(sess.run(m))
    print(sess.run(n))
    print(sess.run(o[1]))#thrid
