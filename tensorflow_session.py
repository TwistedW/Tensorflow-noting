import tensorflow as tf

matrix1 = tf.constant([[5, 3]])
matrix2 = tf.constant([[2], [3]])
product = tf.matmul(matrix1, matrix2) #matrix multiply

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#another method
with tf.Session() as sess:
    print(sess.run(product))
