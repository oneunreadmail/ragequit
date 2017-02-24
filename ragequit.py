import os
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

x_y_train = []
for i in range(len(x_train)):
    x_y_train.append((x_train[i], y_train[i]))


def batch(x_y, size=10):
    s = random.sample(x_y, size)
    return [x_y[0] for x_y in s], [x_y[1] for x_y in s]


def weight_variable(shape, name=None):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, mean=0.5, stddev=0.4)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """Create a bias variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial, name=name)


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


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.sigmoid):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim], name="W_" + layer_name)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim], name="b_" + layer_name)
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations

# suppress warnings from tf and start our session
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.InteractiveSession()

# add input data and output ground truth data
x = tf.placeholder(tf.float32, [None, 12], name="x")
y = tf.placeholder(tf.float32, [None, 2], name="y")

# build a structure of hidden layers
hidden1 = nn_layer(x, 12, 10, 'layer1')
hidden2 = nn_layer(hidden1, 10, 2, 'layer2')
#hidden3 = nn_layer(tf.nn.softplus(hidden2), 100, 100, 'layer3')
#hidden4 = nn_layer(tf.nn.softplus(hidden3), 100, 2, "layer4")

# add our model prediction data
out = hidden2

# calculate loss function and define a train step
loss = tf.losses.mean_squared_error(y, out)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# add some fancy graphs to tensorboard
with tf.name_scope("control_room"):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('h1_average', tf.reduce_mean(hidden1))
    tf.summary.scalar('h2_average', tf.reduce_mean(hidden2))
    variable_summaries(out)
    #tf.summary.histogram('out', out)
merged = tf.summary.merge_all()

# initialise log writer for tensorboard
file_writer = tf.summary.FileWriter('logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/", sess.graph)

# initialize our variables
tf.global_variables_initializer().run()

for i in range(10000):
    batch_xs, batch_ys = batch(x_y_train, size=1000)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 10 == 0:
        file_writer.add_summary(sess.run(merged, feed_dict={x: x_train[:100], y: y_train[:100]}), i)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
a, o, cp = sess.run((accuracy, tf.argmax(out, 1), correct_prediction), feed_dict={x: x_train, y: y_train})
print(a, "\n", o[:10])
np.set_printoptions(suppress=True, precision=3)
print("Out contains {0:.1f}% of Trues".format(np.sum(o)/len(o)*100))

#push_test(sess.run(out, feed_dict={x: x_test}), "y_test.csv")
#push_test(sess.run(out, feed_dict={x: x_train}), "y_train_compare.csv")

#print(sess.run([W, b]))