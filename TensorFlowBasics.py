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
output = sess.run(z, feed_dict={x: 4, y: 2}, name="var1")
print(output)
with tf.variable_scope("scope"):
    var = tf.get_variable("var1", [1])
print(var)


