import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from datetime import datetime

class nnetwork():
    ITERATIONS = 2000
    BATCH_SIZE = 100
    feature_importances_ = None

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

    def prepare_ur_anus(self, anus_size):
        # add input data and output ground truth data
        self.x = tf.placeholder(tf.float32, [None, anus_size], name="x")
        self.y = tf.placeholder(tf.float32, [None, 2], name="y")

        # build a structure of hidden layers
        self.hidden1 = self.nn_layer(self.x, anus_size, 2, 'layer1')
        self.hidden2 = self.nn_layer(self.hidden1, 2, 2, 'layer2')
        # hidden3 = nn_layer(hidden2, 4, 2, 'layer3')
        # hidden4 = nn_layer(tf.nn.softplus(hidden3), 100, 2, "layer4")

        # add our model prediction data
        self.out = self.hidden2

        # calculate loss function and define a train step
        self.loss = tf.losses.log_loss(self.out, self.y, weights=1.0, epsilon=1e-15, scope=None)
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

        # add some fancy graphs to tensorboard
        with tf.name_scope("control_room"):
            tf.summary.scalar('loss', self.loss)
            # tf.summary.scalar('h1_average', tf.reduce_mean(hidden1))
            # tf.summary.scalar('h2_average', tf.reduce_mean(hidden2))
            self.variable_summaries(self.out)
            # tf.summary.histogram('out', out)
        self.merged = tf.summary.merge_all()

        # initialise log writer for tensorboard
        str_i_b = "i=" + str(self.ITERATIONS) + "&bs=" + str(self.BATCH_SIZE)
        str_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.FileWriter('logs/' + str_date + "/", self.sess.graph)

        # initialize our variables

    def fit(self, x, y, *args, **kwargs):
        x = np.array(x, ndmin=2)
        y = np.array(y, ndmin=2)
        self.prepare_ur_anus(x.shape[1])
        tf.global_variables_initializer().run()

        for i in range(self.ITERATIONS):
            batch_xs, batch_ys = self.get_batch(x, y, size=self.BATCH_SIZE, rand=True)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y: batch_ys})
            if i % 10 == 0:
                self.file_writer.add_summary(self.sess.run(self.merged,
                                                           feed_dict={self.x: batch_xs, self.y: batch_ys}), i)

    def predict(self, x):
        return np.around(self.sess.run(self.out, feed_dict={self.x: x}))

    def predict_proba(self, x):
        return self.sess.run(self.out, feed_dict={self.x: x})

    def get_batch(self, x, y, size=100, number=None, rand=False):
        assert (x.shape[0] == y.shape[0]), "x shape " + str(x.shape[0]) + " != y.shape " + str(y.shape[0])
        if random:
            sample = random.sample(range(x.shape[0]), size)
            return x[sample], y[sample]
        elif (number == 0) or number:
            if number + size > x.shape[0] or number < 0:
                raise ValueError("get_batch: out of bounds for array")
            sample = list(range(number, number + size))
            return x[sample], y[sample]
        else:
            raise ValueError("get_batch: function requires either number or random=True")

    def weight_variable(self, shape, name=None):
        """Create a weight variable with appropriate initialization."""
        initial = tf.random_uniform(shape, minval=-0.5, maxval=0.5)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
        """Create a bias variable with appropriate initialization."""
        initial = tf.random_uniform(shape, minval=-0.5, maxval=0.5)
        return tf.Variable(initial, name=name)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            #tf.summary.histogram('histogram', var)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.sigmoid):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        if not act:
            act = lambda x: x
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim], name="W_" + layer_name)
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim], name="b_" + layer_name)
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                #tf.summary.histogram('pre_activations', preactivate)
                activations = act(preactivate, name='activation')
                #tf.summary.histogram('activations', activations)
            return activations

xytte = pd.read_csv("data/xytte.csv")
xytte = xytte[[
    'maxPlayerLevel',
    'numberOfAttemptedLevels',
    'attemptsOnTheHighestLevel',
    'totalNumOfAttempts',
    #'averageNumOfTurnsPerCompletedLevel',
    'doReturnOnLowerLevels',
    'numberOfBoostersUsed',
    'fractionOfUsefullBoosters',
    'totalScore',
    'totalBonusScore',
    'totalStarsCount',
    'numberOfDaysActuallyPlayed',
    'returned']]
x_train = xytte[xytte.returned == xytte.returned].reset_index(drop=True).drop("returned", axis=1)
y_train = xytte[xytte.returned == xytte.returned].reset_index(drop=True)[["returned"]]
x_test  = xytte[xytte.returned != xytte.returned].reset_index(drop=True).drop("returned", axis=1)
y_train["gone"] = 1 - y_train.returned

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
x1, x2, y1, y2 = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

nn = nnetwork()
#print(x_train)
nn.fit(x1, y1)

#print(a.shape)
#print(b.shape)
print("PREDICTIONZ")
print(nn.predict_proba(x2)[:10])
print("log loss: ", log_loss(y2, nn.predict_proba(x2)))
#print(nn.predict_proba(a))
