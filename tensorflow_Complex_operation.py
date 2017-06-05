import tensorflow as tf
import numpy as np

real = np.array([[-4.5, 2.3], [1., -2.]])
imag = np.array([[3.2, 1.25], [-1.5, 2.2]])

Complex = tf.complex(real, imag)
Complex_abs = tf.abs(Complex)
Complex_conj = tf.conj(Complex)
Complex_real = tf.real(Complex)
Complex_imag = tf.imag(Complex)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(Complex), '\n', sess.run(Complex_abs), '\n', sess.run(Complex_conj))
    print(sess.run(Complex_real), '\n', sess.run(Complex_imag))