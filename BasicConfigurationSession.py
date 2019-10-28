import tensorflow as tf
# from tensorflow import global_variables_initializer
#
# with tf.device('/gpu:0'):
#     w = tf.get_variable(name="w", initializer=[.5], dtype=tf.float32)
#     b = tf.get_variable(name="b", initializer=[1.0], dtype=tf.float32)
#     x = tf.placeholder(name = 'x', dtype=tf.float32)
#     y = w * x + b
#
# config = tf.ConfigProto()
# config.log_device_placement =True
#
# with tf.Session(config = config) as tf:
#     tf.run(global_variables_initializer())

with tf.device('/gpu:0'): #choose device
    x = tf.placeholder(tf.float32, name="x", shape=[None, 4])
    w = tf.Variable(tf.random_uniform([4, 10], -1, 1), name="w")
    b = tf.Variable(tf.zeros([10]), name="biases")

output = tf.matmul(x, w) + b

init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(log_device_placement = True))
sess.run(init)

import numpy as np
xrand = np.random.rand(10, 4)

for val in xrand:
    print(sess.run(output, feed_dict={x: xrand}))
