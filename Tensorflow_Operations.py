import tensorflow as tf



a = tf.constant(3)
b = tf.constant(2)
add = tf.add(a, b)
multiply = tf.multiply(a, b)

#to check result of add we need to execute session or Interative Session, and then we can execute eval
tf_session = tf.InteractiveSession()
print(add.eval())
print(multiply.eval())