import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppresses warnings from tensorflow

import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from mlwrap import get_test, get_train, push_test


x_train, y_train = get_train()
x_test = get_test()
for i, x in enumerate(x_train):
    x_train[i] = list(np.array(x) *
                      [1/100, 1/100, 1/100, 1/500, 1/40, 1, 1/100, 1, 1/30000000, 1/25000, 1/500, 1/10])

for i, x in enumerate(x_test):
    x_test[i] = list(np.array(x) *
                     [1/100, 1/100, 1/100, 1/500, 1/40, 1, 1/100, 1, 1/30000000, 1/25000, 1/500, 1/10])

#print(x_train[:10])
x_y_train = []
for i in range(len(x_train)):
    x_y_train.append((x_train[i], y_train[i]))


def batch(x_y, size=10):
    s = random.sample(x_y, size)
    return [x_y[0] for x_y in s], [x_y[1] for x_y in s]


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 12], name="x")
y = tf.placeholder(tf.float32, [None, 1], name="y")
"""
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([12, 128]), name="W1")
    b1 = tf.Variable(tf.random_normal([128]), name="b1")
    #r = tf.nn.softmax(tf.matmul(x, W1) + b1)
    r = tf.nn.softmax(tf.matmul(x, W1) + b1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([128, 1]), name="W2")
    b2 = tf.Variable(tf.random_normal([1]), name="b2")
    m = tf.nn.softmax(tf.matmul(r, W2) + b2)
"""
hidden1 = nn_layer(x, 12, 500, 'layer1')
hidden2 = nn_layer(hidden1, 500, 1, "layer2")

out = hidden2

with tf.name_scope("results"):
    variable_summaries(out)
    variable_summaries(tf.squared_difference(y, out))

cross_entropy = tf.reduce_mean(tf.squared_difference(y, out))
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('out_average', tf.reduce_mean(out))

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=m))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

merged = tf.summary.merge_all()
tf.global_variables_initializer().run()
file_writer = tf.summary.FileWriter('logs/'+ datetime.now().strftime("%Y%m%d-%H%M%S") + "/", sess.graph)

for i in range(1000):
    batch_xs, batch_ys = batch(x_y_train)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
    file_writer.add_summary(summary, i)

tf.summary.scalar('cross_entropy', cross_entropy)

correct_prediction = tf.cast(tf.equal(tf.round(out), y), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
a, o, cp = sess.run((accuracy, tf.round(out), correct_prediction), feed_dict={x: x_train, y: y_train})
print(a, "\n", cp[:10])

push_test(sess.run(out, feed_dict={x: x_test}), "y_test.csv")
push_test(sess.run(out, feed_dict={x: x_train}), "y_train_compare.csv")

#print(sess.run([W, b]))