import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()#initialize variables
#init = tf.global_variables_initializer() #used to high version
sess = tf.Session()
sess.run(init)
for _ in range(3):
    sess.run(update)
    print(sess.run(state))
sess.close()