from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Spark initialization using pyspark
from pyspark import SparkContext
sc = SparkContext('local[*]')

from pyjavaproperties import Properties
properties = Properties()
# To change and read the parameters (hyperparameters) from external file (usage: not need to restart the application while running in dsitributed servers)
properties.load(open('/home/giri/global.properties'))

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import time
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data

"""

Title: Training Neural Networks  with ADMM on Apache Spark
Course: Signal Processing for Big Data
Professor's: Sergio Barbarossa and Paolo Di Lorenzo
Student: Dastagiri Dudekula 1826239
University: Sapienza University of Rome
Dept: M.Sc. in Data Science

Project Keywords: ADMM, Neural networks (tensorflow), Apache Spark
"""

class SPARK_ADMM(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, n_train_samples):

        self.a0 = np.zeros((n_inputs, n_train_samples))

        self.w1 = np.zeros((n_hidden, n_inputs))
        self.w2 = np.zeros((n_hidden, n_hidden))
        self.w3 = np.zeros((n_outputs, n_hidden))

        self.z1 = np.random.rand(n_hidden, n_train_samples)
        self.a1 = np.random.rand(n_hidden, n_train_samples)

        self.z2 = np.random.rand(n_hidden, n_train_samples)
        self.a2 = np.random.rand(n_hidden, n_train_samples)

        self.z3 = np.random.rand(n_outputs, n_train_samples)

        self.lambda_lagrange = np.ones((n_outputs, n_train_samples))

    def warming(self, inputs, outputs, epochs, beta, gamma, h_func):

        self.a0 = inputs
        for i in range(epochs):
            time_start = int(round(time.time() * 1000))
            print("Warming epoch %d/%d : " % (i + 1, epochs))

            # Input layer
            self.w1 = self.weight_update(self.z1, self.a0)
            self.a1 = self.activation_update(self.w2, self.z2, self.z1, beta, gamma, h_func)
            self.z1 = self.argminz(self.a1, self.w1, self.a0, beta, gamma, h_func)

            # Hidden layer
            self.w2 = self.weight_update(self.z2, self.a1)
            self.a2 = self.activation_update(self.w3, self.z3, self.z2, beta, gamma, h_func)
            self.z2 = self.argminz(self.a2, self.w2, self.a1, beta, gamma, h_func)

            # Output layer
            self.w3 = self.weight_update(self.z3, self.a2)
            self.z3 = self.argminlastz(outputs, self.lambda_lagrange, self.w3, self.a2, beta)

            print("%d secs" % (((int(round(time.time() * 1000))) - time_start) / 1000))

    def weight_update(self, layer_output, activation_input):

        # pseudo-inverse of input activation
        p_inv = np.linalg.pinv(activation_input)
        # output
        updated_weight_matrix = tf.matmul(tf.cast(layer_output, tf.float32), tf.cast(p_inv, tf.float32))

        return updated_weight_matrix

    def activation_update(self, next_weight, next_layer_output, layer_nl_output, beta, gamma, h_func):

        # Calculate ReLU/Sigmoid
        layer_nl_output = self.h_function(layer_nl_output, h_func)

        # Activation inverse
        m1 = beta * tf.matmul(tf.matrix_transpose(next_weight), next_weight)
        m2 = tf.scalar_mul(gamma, tf.eye(tf.cast(m1.get_shape()[0], tf.int32)))

        # Activation formulate
        m3 = beta * tf.matmul(tf.matrix_transpose(next_weight), next_layer_output)
        m4 = gamma * layer_nl_output

        # Output
        updated_activation_matrix = tf.matmul(tf.matrix_inverse(tf.cast(m1, tf.float32) + tf.cast(m2, tf.float32)), tf.cast(m3, tf.float32) + tf.cast(m4, tf.float32))

        return updated_activation_matrix

    def h_function(self, x, choice=1):

        if choice == 1:
            # Relu activation function
            return tf.maximum(0.,x)
        else:
            # non-differential Sigmoid function
            return tf.nn.sigmoid(x)

    def argminz(self, a, w, a_in, beta, gamma, h_func):

        # output layer matrix Z_l minimization
        w_mul_a = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))

        m1 = np.array(gamma * a + beta * w_mul_a) / (gamma + beta)
        m2 = np.array(w_mul_a)

        z1 = np.zeros_like(a)
        z2 = np.zeros_like(a)
        z_output = np.zeros_like(a)

        z1[m1 >= 0.] = m1[m1 >= 0.]
        z2[m2 <= 0.] = m2[m2 <= 0.]

        fz_1 = gamma * tf.square(a - self.h_function(z1, h_func)) + beta * (tf.square(z1 - w_mul_a))
        fz_2 = gamma * tf.square(a - self.h_function(z2, h_func)) + beta * (tf.square(z2 - w_mul_a))

        fz_1 = np.array(fz_1)
        fz_2 = np.array(fz_2)

        index_z1 = fz_1 <= fz_2
        index_z2 = fz_2 < fz_1

        z_output[index_z1] = z1[index_z1]
        z_output[index_z2] = z2[index_z2]

        return z_output

    def argminlastz(self, targets, lambd, w, a_in, beta):

        # last output layer matrix Z_L+1
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        z = (targets - lambd + beta * m) / (1 + beta)
        return z


    def fit(self, inputs, labels, beta, gamma, h_func):
        # Train
        self.a0 = inputs

        # Input layer
        self.w1 = self.weight_update(self.z1, self.a0)
        self.a1 = self.activation_update(self.w2, self.z2, self.z1, beta, gamma, h_func)
        self.z1 = self.argminz(self.a1, self.w1, self.a0, beta, gamma, h_func)

        # Hidden layer
        self.w2 = self.weight_update(self.z2, self.a1)
        self.a2 = self.activation_update(self.w3, self.z3, self.z2, beta, gamma, h_func)
        self.z2 = self.argminz(self.a2, self.w2, self.a1, beta, gamma, h_func)

        # Output layer
        self.w3 = self.weight_update(self.z3, self.a2)
        self.z3 = self.argminlastz(labels, self.lambda_lagrange, self.w3, self.a2, beta)
        self.lambda_lagrange = self.lambda_update(self.z3, self.w3, self.a2, beta)

        loss, accuracy = self.evaluate(inputs, labels, h_func)
        return loss, accuracy

    def lambda_update(self, last_layer_output_ZL, w, a_in, beta):

        w_mul_a = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))

        update_lambda = beta * (last_layer_output_ZL - w_mul_a)

        return update_lambda

    def evaluate(self, inputs, target, h_func):
        # Classification
        predicted = self.feed_forward(inputs, h_func)
        loss = tf.reduce_mean(tf.square(predicted - target))

        accuracy = tf.equal(tf.argmax(target, axis=0), tf.argmax(predicted, axis=0))
        accuracy = tf.reduce_sum(tf.cast(accuracy, tf.int32)) / accuracy.get_shape()[0]

        return loss, accuracy

    def feed_forward(self, inputs, h_func):
        # Prediction
        outputs = self.h_function(tf.matmul(self.w1, inputs), h_func)
        outputs = self.h_function(tf.matmul(self.w2, outputs), h_func)
        # output
        outputs = tf.matmul(self.w3, outputs)
        return outputs

    def drawcurve(self, tr, v, ts, id, epoch, name):

        star = mpath.Path.unit_regular_star(6)
        circle = mpath.Path.unit_circle()
        # concatenate the circle with an internal cutout of the star
        verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
        codes = np.concatenate([circle.codes, star.codes])
        cut_star = mpath.Path(verts, codes)

        tr = np.array(tr).flatten()
        v = np.array(v).flatten()
        ts = np.array(ts).flatten()

        plt.figure(id)
        plt.plot(tr, marker=cut_star, markersize=10, color='blue')
        plt.plot(v, marker=cut_star, markersize=10, color='green')
        plt.plot(ts, marker=cut_star, markersize=10, color = 'red')
        plt.savefig(properties.getProperty('PLOTS_SAVE_HERE') + '%s_after_%d_epoch.png' % (name, epoch), dpi=300)
        #plt.draw()
        #plt.pause(0.001)
        return 0

def main():

    try:
        # Extract MNIST data
        mnist = input_data.read_data_sets("./data/", one_hot=True)

        # Paralleize with partitions
        trainX_rdd = sc.parallelize(np.transpose(mnist.train.images).astype(np.float32),
                                    int(properties.getProperty('PARTITIONS'))).cache()
        trainY_rdd = sc.parallelize(np.transpose(mnist.train.labels).astype(np.float32),
                                    int(properties.getProperty('PARTITIONS'))).cache()

        validX_rdd = sc.parallelize(np.transpose(mnist.validation.images).astype(np.float32),
                                    int(properties.getProperty('PARTITIONS'))).cache()
        validY_rdd = sc.parallelize(np.transpose(mnist.validation.labels).astype(np.float32),
                                    int(properties.getProperty('PARTITIONS'))).cache()

        testX_rdd = sc.parallelize(np.transpose(mnist.test.images).astype(np.float32),
                                   int(properties.getProperty('PARTITIONS'))).cache()
        testY_rdd = sc.parallelize(np.transpose(mnist.test.labels).astype(np.float32),
                                   int(properties.getProperty('PARTITIONS'))).cache()

        # model
        model = SPARK_ADMM(int(properties.getProperty('N_INPUTS')), int(properties.getProperty('N_HIDDEN')),
                              int(properties.getProperty('N_OUTPUTS')), int(properties.getProperty('N_TRAIN_BATCHES')))

        # warming Model
        model.warming(np.array(trainX_rdd.collect()), np.array(trainY_rdd.collect()), int(properties.getProperty('EPOCHS_WARNING')), float(properties.getProperty('BETA')), float(properties.getProperty('GAMMA')), int(properties.getProperty('H_FUNCTION')))

        train_losses, valid_losses, train_accuracies, valid_accuracies, test_losses, test_accuracies = ([] for _ in range(6))

        epochs_training = int(properties.getProperty('EPOCHS_TRAINING'))
        best_loss = np.array(float(properties.getProperty('BEST_LOSS')))
        max_calls = int(properties.getProperty('CALLS_TO_STOP'))

        for i in range(epochs_training):
            time_start = int(round(time.time() * 1000))
            print("Epoch %d/%d :: training..." % (i + 1, epochs_training))

            # Train ADMM by minimizing the sub-problems
            train_loss, train_acc = model.fit(np.array(trainX_rdd.collect()), np.array(trainY_rdd.collect()),
                                              float(properties.getProperty('BETA')),
                                              float(properties.getProperty('GAMMA')), int(properties.getProperty('H_FUNCTION')))
            # Classification by feed forward
            valid_loss, valid_acc = model.evaluate(np.array(validX_rdd.collect()), np.array(validY_rdd.collect()), int(properties.getProperty('H_FUNCTION')))

            print("%d secs" % (((int(round(time.time() * 1000))) - time_start) / 1000))
            print(" Train:: Loss = %3f, acc = %3f" % (np.array(train_loss), np.array(train_acc)))
            print(" Valid:: Loss = %3f, acc = %3f" % (np.array(valid_loss), np.array(valid_acc)))

            train_losses.append(np.array(train_loss))
            valid_losses.append(np.array(valid_loss))
            train_accuracies.append(np.array(train_acc))
            valid_accuracies.append(np.array(valid_acc))

            # Testing
            print("Test Results:: ")
            # Classification by feed forward
            test_loss, test_acc = model.evaluate(np.array(testX_rdd.collect()), np.array(testY_rdd.collect()), int(properties.getProperty('H_FUNCTION')))
            print("Loss = %3f, acc = %3f" % (np.array(test_loss), np.array(test_acc)))

            test_losses.append(np.array(test_loss))
            test_accuracies.append(np.array(test_acc))

            model.drawcurve(train_losses, valid_losses, test_losses, 1, i + 1, "LOSS")
            model.drawcurve(train_accuracies, valid_accuracies, test_accuracies, 2, i + 1, "ACC")

            valid_loss = np.array(valid_loss)

            # Stop
            if best_loss > valid_loss:
                best_loss = valid_loss
                calls = 0
            else:
                calls += 1
            if calls > max_calls:
                break


        '''
        # Test results
        print("Test Results:: ")
        test_loss, test_acc = model.evaluate(np.array(testX_rdd.collect()), np.array(testY_rdd.collect()))
        print("Loss = %3f, acc = %3f" % (np.array(test_loss), np.array(test_acc)))

        '''
    finally:
        # clean the chache
        trainX_rdd.unpersist()
        trainY_rdd.unpersist()
        validX_rdd.unpersist()
        validY_rdd.unpersist()
        testX_rdd.unpersist()
        testY_rdd.unpersist()

        # clean the memory by nullifying and deleting
        trainX_rdd = None
        trainY_rdd = None
        validX_rdd = None
        validY_rdd = None
        testX_rdd = None
        testY_rdd = None

        del trainX_rdd
        del trainY_rdd
        del validX_rdd
        del validY_rdd
        del testX_rdd
        del testY_rdd

        # stop the spark context
        sc.stop()


if __name__ == "__main__":
    main()


''' global.properties (reference)
## ADMM_NN_SPARK

PARTITIONS=400
# MNIST image shape 28*28
N_INPUTS=784
# Number of hidden units (neurons)
N_HIDDEN=256
# MNIST classes from 0-9 digits
N_OUTPUTS=10
# Number of data sample that you want to train (MNIST : 55000 number of samples for training)
N_TRAIN_BATCHES=55000

# Priors
BETA=5.0
GAMMA=5.0

EPOCHS_TRAINING=20
EPOCHS_WARNING=10

# plot saving location
PLOTS_SAVE_HERE=/home/giri/SPBD/research/

# 
BEST_LOSS=9999.0
CALLS_TO_STOP=5

# if 1 = ReLU else sigmoid
H_FUNCTION=1

'''