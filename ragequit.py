import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppresses warnings from tensorflow

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
print(sess.run(hello))
print(sess.run([node1, node2]))
print(sess.run(tf.add(node1, node2)))
print(sess.run(ph1 + ph2, {ph1: 1, ph2: 2}))

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(model, {x:[1,2,3,4]}))
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))