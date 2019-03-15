from __future__ import division, print_function

import os
import sys
import copy
import math
import time
import scipy.io
from datetime import datetime
import math
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
from sklearn.model_selection import StratifiedShuffleSplit

__author__ = "Konidaris Vissarion, Logothetis Fragoulis"

np.set_printoptions(threshold=np.nan)
from lib.precision import _FLOATX


class CNN(object):

    def __init__(self, patient, model_id, seed=47, train=True, ld_aug_labels=False):

        self.seed = seed
        self.Patient = patient
        self.model_id = model_id
        self.model_path = "C:\\Users\\Username\\Desktop\\Epilepsy\\code\\saved_models\\DeepParams\\Patient_" + \
                          str(patient) + "\\Model_" + str(self.model_id) + "\\cnn.ckpt"
        self.ld_aug_labels = ld_aug_labels
        self.trainable = train

        self.decay_factor = 1.0005 # 1.00035
        self.lr0 = 0.05
        self.l_min = 1e-6
        self.learning_rate = copy.deepcopy(self.lr0)
        self.data_fitted = 0
        self.batch_sz = 64

        # Gathering stats
        self.stat_train_loss = []
        self.stat_valid_loss = []
        self.stat_valid_tp = []
        self.stat_valid_fp = []
        self.stat_valid_fn = []
        self.stat_valid_f1_score = []
        self.stat_test_tp = []
        self.stat_test_fp = []
        self.stat_test_fn = []
        self.stat_test_accuracy = []
        self.stat_test_f1_score = []

        np.random.seed(self.seed)
        self.define_net()

    def define_DataSource(self):
        # --- The data sources
        file_prefix = "C:\\Users\\Username\\Desktop\\Epilepsy\\data\\sp\\Patient_" + self.Patient + "\\"
        path, dirs, files = next(os.walk(file_prefix))
        self.dataset_length = len(files)

        # We know that images have 128 pixels in each dimension.
        spectro_size = 128

        # Images are stored in one-dimensional arrays of this length.
        spectro_size_flat = spectro_size * spectro_size

        ##__________________________Read Input__________________________
        data_images = []
        labels = []

        for num_data in range(self.dataset_length):

            with open(str(file_prefix + files[num_data]), 'rb') as fid:
                data = np.fromfile(fid, dtype=np.float32, count=-1)
                data = data.reshape((int(len(data) / spectro_size_flat), spectro_size, spectro_size))
                data = np.transpose(data, [2, 1, 0])
                data_images.append(data)
                del data
            if (files[num_data][11] == 'n'):
                labels.append([0, 1])
            else:
                labels.append([1, 0])
            fid.close()

        data_images = np.array(data_images, dtype=np.float32)
        data_images = np.log10(data_images)
        labels = np.array(labels, dtype=np.float32)
        aug_labels = np.zeros((self.dataset_length, 1), dtype=np.float32)

        # Shuffling the dataset at random.
        random_suffle = np.random.choice(self.dataset_length, self.dataset_length, replace=False)
        data_images = data_images[random_suffle]
        labels = labels[random_suffle]

        # The number of channels of the spectrogram image.
        self.channels = int(np.shape(data_images)[3])

        self.epileptic_seizures = 0
        for i in range(self.dataset_length):
            if (labels[i, 0] == 1):
                self.epileptic_seizures += 1

        print('Number of epileptic seizures in the data: ', self.epileptic_seizures)
        print('Number of normal brain activity in the data: ', self.dataset_length - self.epileptic_seizures)
        print('Data Were Successfully Loaded')
        print('')

        # Create the augmented labels in case of transfer learning.
        if (self.ld_aug_labels):
            aug_file_prefix = "C:\\Users\\Username\\Desktop\\Epilepsy\\data\\Patient_" + self.Patient + "\\"
            for num_data in range(1, self.dataset_length + 1):
                Ictal = True
                data_file_name = aug_file_prefix + "Patient_" + str(self.Patient) + "_"
                if (num_data <= self.epileptic_seizures):
                    data_file_name += "ictal_segment_"
                    data_file_name += str(num_data)
                else:
                    Ictal = False
                    data_file_name += "interictal_segment_"
                    data_file_name += str(num_data - self.epileptic_seizures)
                print(data_file_name)
                mat_data = scipy.io.loadmat(data_file_name)
                if (Ictal):
                    if (mat_data['latency'] <= 14.):
                        aug_labels[num_data - 1, 0] = 0.
                    else:
                        aug_labels[num_data - 1, 0] = 1.
                else:
                    aug_labels[num_data - 1, 0] = 2.

            aug_labels = aug_labels[random_suffle]

        ##______________________Creating the training, validation and test set____________________
        stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=self.seed)
        for train_idx, test_idx in stratSplit.split(data_images, labels):
            x_train, self.X_test = data_images[train_idx], data_images[test_idx]
            y_train, self.Y_test = labels[train_idx], labels[test_idx]
            ay_train, self.aY_test = aug_labels[train_idx], aug_labels[test_idx]
        del data_images, labels, aug_labels
        for train_idx, test_idx in stratSplit.split(x_train, y_train):
            self.X_train, self.X_val = x_train[train_idx], x_train[test_idx]
            self.Y_train, self.Y_val = y_train[train_idx], y_train[test_idx]
            self.aY_train, self.aY_val = ay_train[train_idx], ay_train[test_idx]
        del x_train, y_train, ay_train

        self.training_set_size = np.shape(self.Y_train)[0]
        self.validation_set_size = np.shape(self.Y_val)[0]
        self.test_set_size = np.shape(self.Y_test)[0]

    def inference(self):
        # --- Network Architecture
        with tf.device('/gpu:0'):
            with tf.name_scope('Convolution_Weights'):
                # --- Trainable Convolution variables
                self.CW1 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, self.channels, 72],
                                           dtype=tf.float32, name='conv_weights_1')
                self.CW2 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 72, 100],
                                           dtype=tf.float32, name='conv_weights_2')
                self.CW3 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 100, 128],
                                           dtype=tf.float32, name='conv_weights_3')
                self.CW4 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 128, 150],
                                           dtype=tf.float32, name='conv_weights_4')
                self.CW5 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 150, 220],
                                           dtype=tf.float32, name='conv_weights_5')
                self.CW6 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 220, 256],
                                           dtype=tf.float32, name='conv_weights_6')

            with tf.name_scope('Dense_Layer_Weights'):
                # --- Trainable Variables
                self.W1 = tf.get_variable(initializer=xavier_initializer(), shape=[2 * 2 * 256, 256],
                                          dtype=tf.float32,
                                          name='weights_1')
                self.W2 = tf.get_variable(initializer=xavier_initializer(), shape=[256, 512], dtype=tf.float32,
                                          name='weights_2')
                self.W3 = tf.get_variable(initializer=xavier_initializer(), shape=[512, 2], dtype=tf.float32,
                                          name='weights_3')
                self.b = tf.get_variable(initializer=xavier_initializer(), shape=[2], dtype=tf.float32,
                                         name='biases')

            with tf.name_scope('Network'):
                # --- Convolutional Layer 1
                conv1 = tf.nn.conv2d(input=self.X, filter=self.CW1, strides=[1, 1, 1, 1], padding="SAME",
                                     name="conv1")
                b_conv1 = layers.batch_normalization(inputs=conv1, training=self.phase, name="b_conv1")
                a1 = tf.nn.elu(b_conv1, name="a1")
                pool1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")

                # --- Convolutional Layer 2
                conv2 = tf.nn.conv2d(input=pool1, filter=self.CW2, strides=[1, 1, 1, 1], padding="SAME",
                                     name="conv2")
                b_conv2 = layers.batch_normalization(inputs=conv2, training=self.phase, name="b_conv2")
                a2 = tf.nn.elu(b_conv2, name="a2")
                pool2 = tf.nn.max_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")
                dropout1 = tf.nn.dropout(pool2, keep_prob=self.prob, name="dropout1")

                # --- Convolutional Layer 3
                conv3 = tf.nn.conv2d(input=dropout1, filter=self.CW3, strides=[1, 1, 1, 1], padding="SAME",
                                     name="conv3")
                b_conv3 = layers.batch_normalization(inputs=conv3, training=self.phase, name="b_conv3")
                a3 = tf.nn.elu(b_conv3, name="a3")
                pool3 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool3")
                dropout2 = tf.nn.dropout(pool3, keep_prob=self.prob, name="dropout2")

                # --- Convolutional Layer 4
                conv4 = tf.nn.conv2d(input=dropout2, filter=self.CW4, strides=[1, 1, 1, 1], padding="SAME",
                                     name="conv4")
                b_conv4 = layers.batch_normalization(inputs=conv4, training=self.phase, name="b_conv4")
                a4 = tf.nn.elu(b_conv4, name="a4")
                pool4 = tf.nn.max_pool(a4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool4")
                dropout3 = tf.nn.dropout(pool4, keep_prob=self.prob, name="dropout3")

                # --- Convolutional Layer 5
                conv5 = tf.nn.conv2d(input=dropout3, filter=self.CW5, strides=[1, 1, 1, 1], padding="SAME",
                                     name="conv5")
                b_conv5 = layers.batch_normalization(inputs=conv5, training=self.phase, name="b_conv5")
                a5 = tf.nn.elu(b_conv5, name="a5")
                pool5 = tf.nn.max_pool(a5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool5")
                dropout4 = tf.nn.dropout(pool5, keep_prob=self.prob, name="dropout4")

                # --- Convolutional Layer 6
                conv6 = tf.nn.conv2d(input=dropout4, filter=self.CW6, strides=[1, 1, 1, 1], padding="SAME",
                                     name="conv6")
                b_conv6 = layers.batch_normalization(inputs=conv6, training=self.phase, name="b_conv6")
                a6 = tf.nn.elu(b_conv6, name="a6")
                self.pool6 = tf.nn.max_pool(a6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool6")
                dropout5 = tf.nn.dropout(self.pool6, keep_prob=self.prob, name="dropout5")

                # --- Hidden layer 1
                layer1 = tf.matmul(tf.reshape(dropout5, [-1, 2 * 2 * 256]), self.W1, name="layer1")
                b_layer1 = layers.batch_normalization(layer1, training=self.phase, name="b_layer1")
                a7 = tf.nn.elu(b_layer1, name="a7")
                dropout6 = tf.nn.dropout(a7, keep_prob=self.prob, name="dropout6")

                # --- Hidden layer 2
                layer2 = tf.matmul(dropout6, self.W2, name="layer2")
                b_layer2 = layers.batch_normalization(layer2, training=self.phase, name="b_layer2")
                a8 = tf.nn.elu(b_layer2, name="a8")
                dropout7 = tf.nn.dropout(a8, keep_prob=self.prob, name="dropout7")

                # --- Output layer
                self.logits = tf.add(tf.matmul(dropout7, self.W3), self.b, name="logits")

        return self.logits

    def define_train_operations(self):
        # --- Train computations
        with tf.device('/gpu:0'):
            # --- Loss function
            with tf.name_scope('loss'):
                regularization = self.weight_decay_coef * ((tf.norm(self.W1, ord=2) ** 2 / 2) \
                                                           + (tf.norm(self.W2, ord=2) ** 2 / 2) \
                                                           + (tf.norm(self.W3, ord=2) ** 2 / 2))
                scaled_logits_max = tf.multiply(tf.maximum(self.logits, 0.), self.prior_prob)
                scaled_logits_min = tf.multiply(tf.minimum(self.logits, 0.), 2.-self.prior_prob)
                scaled_logits = scaled_logits_max + scaled_logits_min
                # scaled_logits = tf.multiply(self.logits, self.prior_prob)
                cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=scaled_logits, targets=self.Y,
                                                                         pos_weight=0.8)
                self.loss = tf.add(tf.reduce_mean(cross_entropy), regularization, name='loss')

            # --- Optimization
            with tf.name_scope('train'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
                    self.train = optimizer.minimize(self.loss)

    def define_predict_operations(self):
        with tf.device('/gpu:0'):
            # --- Predictions
            with tf.name_scope('Predictions'):
                self.pred = tf.nn.softmax(self.logits, name='pred')
                self.pred_cls = tf.argmax(self.pred, axis=1, name='pred_cls')

            # --- Performance Measures
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(self.pred_cls, tf.argmax(self.Y, axis=1))
                self.corrects = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def initialize_net_inputs(self):
        with tf.device('/gpu:0'):
            with tf.name_scope('Inputs'):
                # --- Auxiliary terms
                self.X = tf.placeholder(tf.float32, shape=[None, 128, 128, self.channels], name='x')
                self.Y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
                self.weight_decay_coef = tf.constant(0.00001, dtype=tf.float32, name='L2_Regularization_Weight')
                self.prior_prob = tf.constant(
                    [[self.epileptic_seizures / self.dataset_length, \
                      (self.dataset_length - self.epileptic_seizures) / self.dataset_length]
                     ], dtype=tf.float32, name='Prior_Class_Probabilities')
                self.prob = tf.placeholder_with_default(1.0, shape=(), name='hold_probability')
                self.phase = tf.placeholder_with_default(False, shape=(), name='training_phase')
                self.lr = tf.placeholder(tf.float32, name='Learning_Rate')

    def define_net(self):
        self.define_DataSource()
        self.initialize_net_inputs()
        self.inference()
        self.define_train_operations()
        self.define_predict_operations()

    def init_Variables(self, sess):
        sess.run(tf.group(tf.global_variables_initializer()))
        self.saver = tf.train.Saver(max_to_keep=4)
        print("Variables initilialized.")

    def restore_variables(self, sess):
        self.saver.restore(sess, self.model_path)
        print("Pretrained variables restored.")

    def ConvOut(self, sess):

        aug_train = np.zeros((self.training_set_size, 2 * 2 * 256), dtype=np.float32)
        aug_valid = np.zeros((self.validation_set_size, 2 * 2 * 256), dtype=np.float32)
        aug_test = np.zeros((self.test_set_size, 2 * 2 * 256), dtype=np.float32)

        # Creating the augmented training dataset.
        pointer = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + self.batch_sz > self.training_set_size):
                batch = self.training_set_size % self.batch_sz
            else:
                batch = self.batch_sz

            conv_out = sess.run(self.pool6, feed_dict={self.X: self.X_train[pointer:pointer + batch]})
            aug_train[pointer:pointer + batch] = np.array(conv_out).reshape((batch, 2 * 2 * 256))
            # Epoch termination condition.
            pointer += batch
            if (pointer == self.training_set_size):
                break

        # Creating the augmented validation dataset.
        pointer = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + self.batch_sz > self.validation_set_size):
                batch = self.validation_set_size % self.batch_sz
            else:
                batch = self.batch_sz

            conv_out = sess.run(self.pool6, feed_dict={self.X: self.X_val[pointer:pointer + batch]})
            aug_valid[pointer:pointer + batch] = np.array(conv_out).reshape((batch, 2 * 2 * 256))

            # Epoch termination condition.
            pointer += batch
            if (pointer == self.validation_set_size):
                break

        # Creating the augmented test dataset.
        pointer = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + self.batch_sz > self.test_set_size):
                batch = self.test_set_size % self.batch_sz
            else:
                batch = self.batch_sz

            conv_out = sess.run(self.pool6, feed_dict={self.X: self.X_test[pointer:pointer + batch]})
            aug_test[pointer:pointer + batch] = np.array(conv_out).reshape((batch, 2 * 2 * 256))

            # Epoch termination condition.
            pointer += batch
            if (pointer == self.test_set_size):
                break

        aug_train = np.hstack((aug_train, self.aY_train.reshape(aug_train.shape[0], 1)))
        aug_valid = np.hstack((aug_valid, self.aY_val.reshape(aug_valid.shape[0], 1)))
        aug_test = np.hstack((aug_test, self.aY_test.reshape(aug_test.shape[0], 1)))

        return aug_train, aug_valid, aug_test

    def train_epoch(self, sess):
        # Shuffling the training dataset at random.
        random_suffle = np.random.choice(self.training_set_size, self.training_set_size, replace=False)
        X_tr = self.X_train[random_suffle]
        Y_tr = self.Y_train[random_suffle]

        pointer = 0
        total_batches = 0
        train_loss = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + self.batch_sz > self.training_set_size):
                batch = self.training_set_size % self.batch_sz
            else:
                batch = self.batch_sz

            # Training on batch.
            mean_loss, _ = sess.run([self.loss, self.train], feed_dict={self.X: X_tr[pointer:pointer + batch], \
                                                                        self.Y: Y_tr[pointer:pointer + batch], \
                                                                        self.lr: self.learning_rate, self.prob: 0.5, \
                                                                        self.phase: True})
            # Gathering info.
            if math.isnan(mean_loss):
                print('train cost is NaN1')
                break
            train_loss += mean_loss
            total_batches += 1
            pointer += batch
            self.data_fitted += batch

            # Performing exponential learning rate decay
            if (self.learning_rate > self.l_min):
                if (batch == self.batch_sz):
                    self.learning_rate = self.lr0 * (1 / self.decay_factor) ** (self.data_fitted / self.batch_sz)
                    if (self.learning_rate < self.l_min):
                        self.learning_rate = self.l_min

            # Epoch termination condition.
            if (pointer == self.training_set_size):
                break

        if (total_batches > 0):
            train_loss /= total_batches

        # Gathering statistics
        if (self.trainable):
            self.stat_train_loss.append(train_loss)

        return train_loss

    def valid_epoch(self, sess):
        pointer = 0
        total_batches = 0
        valid_loss = 0
        V_f1 = 0
        tp = 0
        fp = 0
        fn = 0
        while True:

            # Calculating the correct batch size
            if (pointer + self.batch_sz > self.validation_set_size):
                batch = self.validation_set_size % self.batch_sz
            else:
                batch = self.batch_sz

            mean_loss = sess.run(self.loss, feed_dict={self.X: self.X_val[pointer:pointer + batch], \
                                                       self.Y: self.Y_val[pointer:pointer + batch]})

            # Calculating the F1 score.
            pred = sess.run(self.pred_cls,
                            feed_dict={self.X: self.X_val[pointer:pointer + batch], \
                                       self.Y: self.Y_val[pointer:pointer + batch]})
            true = np.argmax(self.Y_val[pointer:pointer + batch], axis=1)
            for i in range(batch):
                if (true[i] == 0):
                    if (pred[i] == 0):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if (pred[i] == 0):
                        fp += 1

            # Gathering info.
            if math.isnan(mean_loss):
                print('train cost is NaN3')
                break
            valid_loss += mean_loss
            total_batches += 1
            pointer += batch

            # Epoch termination condition.
            if (pointer == self.validation_set_size):
                break

        # Calculating the F1-Score on the validation set
        if (total_batches > 0):
            valid_loss /= total_batches
            self.stat_valid_loss.append(valid_loss)
        if ((2 * tp + fp + fn) > 0):
            V_f1 = (2 * tp) / (2 * tp + fp + fn)

        # Gathering statistics
        if (self.trainable):
            self.stat_valid_tp.append(tp)
            self.stat_valid_fp.append(fp)
            self.stat_valid_fn.append(fn)
            self.stat_valid_f1_score.append(V_f1)

        return valid_loss, V_f1

    def test_Score(self, sess):
        pointer = 0
        test_accuracy = 0
        f1 = 0
        tp = 0
        fp = 0
        fn = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + self.batch_sz > self.test_set_size):
                batch = self.test_set_size % self.batch_sz
            else:
                batch = self.batch_sz

            acc = sess.run(self.corrects, feed_dict={self.X: self.X_test[pointer:pointer + batch], \
                                                     self.Y: self.Y_test[pointer:pointer + batch]})

            # Calculating the F1 score.
            pred = sess.run(self.pred_cls,
                            feed_dict={self.X: self.X_test[pointer:pointer + batch], \
                                       self.Y: self.Y_test[pointer:pointer + batch]})

            true = np.argmax(self.Y_test[pointer:pointer + batch], axis=1)
            for i in range(batch):
                if (true[i] == 0):
                    if (pred[i] == 0):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if (pred[i] == 0):
                        fp += 1

            test_accuracy += acc
            pointer += batch

            # Epoch termination condition.
            if (pointer == self.test_set_size):
                break

        # Calculating the accuracy and the F1-Score on the training set
        accuracy = test_accuracy / self.test_set_size
        if ((2 * tp + fp + fn) > 0):
            f1 = (2 * tp) / (2 * tp + fp + fn)

        # Gathering statistics
        if (self.trainable):
            self.stat_test_accuracy.append(accuracy)
            self.stat_test_tp.append(tp)
            self.stat_test_fp.append(fp)
            self.stat_test_fn.append(fn)
            self.stat_test_f1_score.append(f1)

        return accuracy, tp, fp, fn, f1

    def fit(self, sess):

        epoch = 0
        min_my_metric = sys.float_info.max
        max_epochs = 1000
        n_early_stop_epochs = 50
        early_stop_counter = 0

        best_epoch = 0
        best_f1_score = 0
        best_accuracy = 0

        log_path = "C:\\Users\\Username\\Desktop\\Epilepsy\\code\\saved_models\\Graphing\\Patient_" + \
                   self.Patient + "\\Model_" + str(self.model_id)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, "log_log10_priors.csv")

        self.init_Variables(sess)
        print("Starting training.")
        print("")

        while (epoch < max_epochs):
            epoch += 1

            # _____________________Train Epoch_______________________
            train_loss = self.train_epoch(sess)

            # ____________________Validation Epoch___________________
            valid_loss, V_f1 = self.valid_epoch(sess)

            # _______________________Accuracy calculation____________
            accuracy, tp, fp, fn, f1 = self.test_Score(sess)

            # ________Saving training statistics for plotting________
            if (epoch - 1 != len(self.stat_train_loss)):
                log = open(log_file, 'a')
                for i in range(epoch - 1, len(self.stat_train_loss)):
                    log.write(str(self.stat_train_loss[i]) + "," + str(self.stat_valid_loss[i]) + "," + \
                              str(self.stat_valid_tp[i]) + "," + str(self.stat_valid_fp[i]) + "," + \
                              str(self.stat_valid_fn[i]) + "," + str(self.stat_valid_f1_score[i]) + "," + \
                              str(self.stat_test_accuracy[i]) + "," + str(self.stat_test_tp[i]) + "," + \
                              str(self.stat_test_fp[i]) + "," + str(self.stat_test_fn[i]) + "," + \
                              str(self.stat_test_f1_score[i]))
                    log.write("\n")
                log.close()

            ##_______________________________Early Stopping______________________________
            my_metric = 0.5 * valid_loss + 0.5 * (1 - V_f1)

            if my_metric < min_my_metric:
                print('Best epoch = ' + str(epoch))
                min_my_metric = my_metric
                early_stop_counter = 0
                save_path = self.saver.save(sess, self.model_path)
                best_epoch = epoch
                best_f1_score = f1
                best_accuracy = accuracy
            else:
                early_stop_counter += 1

            ##______________________________Print Epoch Info____________________________
            print('Epoch : ', epoch)
            print('Early Stopping : ' + str(early_stop_counter) + "/" + str(n_early_stop_epochs))
            print('Learning Rate : ', self.learning_rate)
            print('Train : ', train_loss)
            print('Valid : ', valid_loss)
            print('Accuracy : ', accuracy)
            print('Valid F1-score : ', V_f1)
            print('####################################')
            print('TP : ', tp)
            print('FP : ', fp)
            print('FN : ', fn)
            print('F1-score : ', f1)
            print('####################################')
            print('')

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                break

        print('Best epoch : ', best_epoch)
        print('Best accuracy : ', best_accuracy)
        print('Best F1-score : ', best_f1_score)
