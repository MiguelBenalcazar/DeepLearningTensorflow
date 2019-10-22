import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#Starting Sessions:

deep_learning = tf.constant("Deep Learning wit tensorFLow")
session = tf.Session()
session.run(deep_learning)
print(session.run(deep_learning))
session.close()

#Interactive Session
tf_session = tf.InteractiveSession()
print(deep_learning.eval())

weights = tf.Variable(tf.random_normal([5, 10], stddev= 0.5), name = "weights")
x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")
z = tf.multiply(x, y)


sess = tf.Session()
output = sess.run(z, feed_dict={x: 4, y: 2})
print(output)

var = tf.Variable(tf.random_normal([2,3]),name="var1")
with tf.variable_scope("scope"):
    var = tf.get_variable("var1",[1])
print(var.name)

#set the reuse flag to True
with tf.variable_scope("scope1"):
    #declare a variable named var3
    var1 = tf.get_variable("var2", [1])
    #set reuse flag to True
    tf.get_variable_scope().reuse_variables()
    #just an assertion!
    assert tf.get_variable_scope().reuse == True
    #declare another variable with same name
    var2 = tf.get_variable("var2", [1])

print(var1, var2)

#creating some tensors
tf_0 = tf.convert_to_tensor(8.0, dtype=tf.float64)

print(tf_0.eval())

#Convert from numpy to tensor
import numpy as np
vector_id = np.array([4, 5, 6, 0, 7])
tf_ID = tf.convert_to_tensor(vector_id, dtype=tf.float64)
print(tf_ID.eval())

vector_id_2 = np.linspace(0,15,100)
tf_ID_2 = tf.convert_to_tensor(vector_id_2, dtype=tf.int64)
print(tf_ID_2.eval())

#Configure sessions:

with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, name="x", shape=[None, 4])
    W = tf.Variable(tf.random_uniform([4,10], -1, 1), name="W")
    b = tf.Variable(tf.zeros([10]), name="biases")

output = tf.matmul(x, W) + b
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

import numpy as np
xrand = np.random.rand(10, 4)
xrand[:5]

for val in xrand:
    print(sess.run(output, feed_dict={x: xrand}))