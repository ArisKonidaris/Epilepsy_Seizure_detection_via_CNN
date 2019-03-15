import os
import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow import layers
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
from lib.model_io import save_variables
np.set_printoptions(threshold=np.nan)

Patient = "7"
file_prefix = "C:\\Users\\Username\\Desktop\\Epilepsy\\data\\sp\\Patient_"+Patient+"\\"
path, dirs, files = next(os.walk(file_prefix))
dataset_length = len(files)

# We know that images have 128 pixels in each dimension.
spectro_size = 128

# Images are stored in one-dimensional arrays of this length.
spectro_size_flat = spectro_size * spectro_size

##__________________________Read Input__________________________
data_images = []
labels = []

for num_data in range(dataset_length):

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
labels = np.array(labels, dtype=np.float32)

# The number of channels of the spectrogram image.
channels = int(np.shape(data_images)[3])

spectro_image_flat = spectro_size_flat * channels

# Tuple with height and width of images used to reshape arrays.
channel_spectro_shape = (spectro_size, spectro_size)
specto_shape = (spectro_size, spectro_size, channels)

# Shuffling the training dataset at random.
random_suffle = np.random.choice(dataset_length, dataset_length, replace=False)
data_images = data_images[random_suffle]
data_images = np.log10(data_images)
labels = labels[random_suffle]

epileptic_seizures = 0
for i in range(dataset_length):
    if(labels[i,0]==1):
        epileptic_seizures +=1

print('Number of epileptic seizures in the data: ', epileptic_seizures)
print('Number of normal brain activity in the data: ', dataset_length-epileptic_seizures)
print('Data Was Successfully Loaded')
print('')

##______________________Create the training, validation and test set____________________
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in stratSplit.split(data_images, labels):
    x_train, X_test = data_images[train_idx], data_images[test_idx]
    y_train, Y_test = labels[train_idx], labels[test_idx]
del data_images,labels
for train_idx, test_idx in stratSplit.split(x_train, y_train):
    X_train, X_val = x_train[train_idx], x_train[test_idx]
    Y_train, Y_val = y_train[train_idx], y_train[test_idx]
del x_train, y_train

training_set_size = np.shape(Y_train)[0]
validation_set_size = np.shape(Y_val)[0]
test_set_size = np.shape(Y_test)[0]

##_________________________CNN Implementation___________________
weight_decay_coef = 0.00001
with tf.device('/gpu:0'):

    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, 128, 128, channels], name='x')
        Y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
        X_v = tf.placeholder(tf.float32, shape=[None, 128, 128, channels], name='x')
        Y_v = tf.placeholder(tf.float32, shape=[None, 2], name='y')
        prior_prob = tf.constant([[epileptic_seizures/dataset_length, (dataset_length-epileptic_seizures)/dataset_length]], dtype=tf.float32, name='Prior_Class_Probabilities')
        prob = tf.placeholder_with_default(1.0, shape=(), name='hold_probability')
        phase = tf.placeholder_with_default(False, shape=(), name='training_phase')
        lr = tf.placeholder(tf.float32, name='Learning_Rate')

    with tf.name_scope('Convolution_Weights'):
        # Variables to be learned by the model
        CW1 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, channels, 72], dtype=tf.float32, name='conv_weights_1')
        CW2 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 72, 100], dtype=tf.float32, name='conv_weights_2')
        CW3 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 100, 128], dtype=tf.float32, name='conv_weights_3')
        CW4 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 128, 150], dtype=tf.float32, name='conv_weights_4')
        CW5 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 150, 220], dtype=tf.float32, name='conv_weights_5')
        CW6 = tf.get_variable(initializer=xavier_initializer_conv2d(), shape=[3, 3, 220, 256], dtype=tf.float32, name='conv_weights_6')

    with tf.name_scope('Dense_Layer_Weights'):
        W1 = tf.get_variable(initializer=xavier_initializer(), shape=[2 * 2 * 256, 256], dtype=tf.float32, name='weights_1')
        W2 = tf.get_variable(initializer=xavier_initializer(), shape=[256, 512], dtype=tf.float32, name='weights_2')
        W3 = tf.get_variable(initializer=xavier_initializer(), shape=[512, 2], dtype=tf.float32, name='weights_3')
        b = tf.get_variable(initializer=xavier_initializer(), shape=[2], dtype=tf.float32, name='biases')

    with tf.name_scope('Network'):
        # Convolutional Layer 1.
        conv1 = tf.nn.conv2d(input=X, filter=CW1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
        b_conv1 = layers.batch_normalization(inputs=conv1, training=phase, name="b_conv1")
        a1 = tf.nn.elu(b_conv1, name="a1")
        pool1 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")

        # Convolutional Layer 2.
        conv2 = tf.nn.conv2d(input=pool1, filter=CW2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        b_conv2 = layers.batch_normalization(inputs=conv2, training=phase, name="b_conv2")
        a2 = tf.nn.elu(b_conv2, name="a2")
        pool2 = tf.nn.max_pool(a2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")
        dropout1 = tf.nn.dropout(pool2, keep_prob=prob, name="dropout1")

        # Convolutional Layer 3.
        conv3 = tf.nn.conv2d(input=dropout1, filter=CW3, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
        b_conv3 = layers.batch_normalization(inputs=conv3, training=phase, name="b_conv3")
        a3 = tf.nn.elu(b_conv3, name="a3")
        pool3 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool3")
        dropout2 = tf.nn.dropout(pool3, keep_prob=prob, name="dropout2")

        # Convolutional Layer 4.
        conv4 = tf.nn.conv2d(input=dropout2, filter=CW4, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
        b_conv4 = layers.batch_normalization(inputs=conv4, training=phase, name="b_conv4")
        a4 = tf.nn.elu(b_conv4, name="a4")
        pool4 = tf.nn.max_pool(a4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool4")
        dropout3 = tf.nn.dropout(pool4, keep_prob=prob, name="dropout3")

        # Convolutional Layer 5.
        conv5 = tf.nn.conv2d(input=dropout3, filter=CW5, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
        b_conv5 = layers.batch_normalization(inputs=conv5, training=phase, name="b_conv5")
        a5 = tf.nn.elu(b_conv5, name="a5")
        pool5 = tf.nn.max_pool(a5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool5")
        dropout4 = tf.nn.dropout(pool5, keep_prob=prob, name="dropout4")

        # Convolutional Layer 6.
        conv6 = tf.nn.conv2d(input=dropout4, filter=CW6, strides=[1, 1, 1, 1], padding="SAME", name="conv6")
        b_conv6 = layers.batch_normalization(inputs=conv6, training=phase, name="b_conv6")
        a6 = tf.nn.elu(b_conv6, name="a6")
        pool6 = tf.nn.max_pool(a6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool6")
        dropout5 = tf.nn.dropout(pool6, keep_prob=prob, name="dropout5")

        # Hidden layer 1.
        layer1 = tf.matmul(tf.reshape(dropout5, [-1, 2 * 2 * 256]), W1, name="layer1")
        b_layer1 = layers.batch_normalization(layer1, training=phase, name="b_layer1")
        a7 = tf.nn.elu(b_layer1, name="a7")
        dropout6 = tf.nn.dropout(a7, keep_prob=prob, name="dropout6")

        # Hidden layer 2.
        layer2 = tf.matmul(dropout6, W2, name="layer2")
        b_layer2 = layers.batch_normalization(layer2, training=phase, name="b_layer2")
        a8 = tf.nn.elu(b_layer2, name="a8")
        dropout7 = tf.nn.dropout(a8, keep_prob=prob, name="dropout7")

        # Output layer.
        logits = tf.add(tf.matmul(dropout7, W3), b, name="logits")

        #### Predictions ####
        pred = tf.nn.softmax(logits, name='pred')
        pred_cls = tf.argmax(pred, axis=1, name='pred_cls')

    with tf.name_scope('loss'):
        # Loss function
        regularization = weight_decay_coef * ((tf.norm(W1, ord=2)**2/2)\
                                             +(tf.norm(W2, ord=2)**2/2)\
                                             +(tf.norm(W3, ord=2)**2/2))
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        #weight_per_label = tf.transpose(tf.matmul(Y,tf.transpose(1/prior_prob)))
        #cross_entropy = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        scaled_logits = tf.multiply(logits, prior_prob)
        ##cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits, labels=Y)##
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=scaled_logits, targets=Y, pos_weight=0.2)
        loss = tf.add(tf.reduce_mean(cross_entropy), regularization, name='loss')

        # scaled_logits_max = tf.multiply(tf.maximum(logits, 0.), prior_prob)
        # scaled_logits_min = tf.multiply(tf.minimum(logits, 0.), 2-prior_prob)
        # scaled_logits = scaled_logits_max + scaled_logits_min
        # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=scaled_logits, targets=Y, pos_weight=0.2)
        # loss = tf.add(tf.reduce_mean(cross_entropy), regularization, name='loss')


    # Optimization
    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            train = optimizer.minimize(loss)

    # Performance Measures
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(pred_cls, tf.argmax(Y, axis=1))
        corrects = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # create a summary for our cost and accuracy
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session
    merged = tf.summary.merge_all()

##____________________________________Training the Network_________________________________

batch_sz = 64

epoch = 0
min_my_metric = sys.float_info.max
max_epochs = 1000
n_early_stop_epochs = 50
early_stop_counter = 0

decay_factor = 1.0005
lr0 = 0.5
l_min = 1e-6
learning_rate = lr0
data_fitted = 0

accuracy = 0
best_epoch = 0
best_f1_score = 0
best_accuracy = 0

# Gathering stats
stat_train_loss = []
stat_valid_loss = []
stat_valid_tp = []
stat_valid_fp = []
stat_valid_fn = []
stat_valid_f1_score = []
stat_test_tp = []
stat_test_fp = []
stat_test_fn = []
stat_test_accuracy = []
stat_test_f1_score = []

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#with tf.Session() as sess:

    # A log writer
    #valid_writer = tf.summary.FileWriter("C:\\Users\\Username\\Desktop\\Epilepsy\\logs\\valid", sess.graph)
    #test_writer = tf.summary.FileWriter("C:\\Users\\Username\\Desktop\\Epilepsy\\logs\\test", sess.graph)
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)

    sess.run(tf.global_variables_initializer())
    print("Variables initilialized.")
    print("Starting training.")
    print("")

    while (epoch < max_epochs):
        epoch += 1

        # Shuffling the training dataset at random.
        random_suffle = np.random.choice(training_set_size, training_set_size, replace=False)
        X_train = X_train[random_suffle]
        Y_train = Y_train[random_suffle]

        #_______________________Train Epoch_____________________
        pointer = 0
        total_batches = 0
        train_loss = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + batch_sz > training_set_size):
                batch = training_set_size % batch_sz
            else:
                batch = batch_sz

            # Training on batch.
            mean_loss, _ = sess.run([loss, train], feed_dict={X: X_train[pointer:pointer + batch], \
                                                              Y: Y_train[pointer:pointer + batch], \
                                                              lr: learning_rate, prob: 0.5, phase: True })

            # Gathering info.
            if math.isnan(mean_loss):
                print('train cost is NaN1')
                break
            train_loss += mean_loss
            total_batches += 1
            pointer += batch
            data_fitted += batch

            if(learning_rate>l_min):
                if(batch==batch_sz):
                    learning_rate = lr0 * (1 / decay_factor) ** (data_fitted / batch_sz)
                    if (learning_rate < l_min):
                        learning_rate = l_min

            # Epoch termination condition.
            if (pointer == training_set_size):
                break

        if (total_batches > 0):
            train_loss /= total_batches
        stat_train_loss.append(train_loss)

        # _______________________Validation Epoch_____________________
        pointer = 0
        total_batches = 0
        valid_loss = 0
        V_f1 = 0
        tp = 0
        fp = 0
        fn = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + batch_sz > validation_set_size):
                batch = validation_set_size % batch_sz
            else:
                batch = batch_sz

            # Training on batch.
            # summary, mean_loss = sess.run([merged, loss], feed_dict={X: X_val[pointer:pointer + batch], \
            #                                                          Y: Y_val[pointer:pointer + batch] })
            mean_loss = sess.run(loss, feed_dict={X: X_val[pointer:pointer + batch], \
                                                  Y: Y_val[pointer:pointer + batch]})

            # Calculating the F1 score.
            pred = sess.run(pred_cls,
                            feed_dict={X: X_val[pointer:pointer + batch], Y: Y_val[pointer:pointer + batch]})
            true = np.argmax(Y_val[pointer:pointer + batch], axis=1)
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
            #valid_writer.add_summary(summary, epoch*pointer)

            # Epoch termination condition.
            if (pointer == validation_set_size):
                break
        if (total_batches > 0):
            valid_loss /= total_batches
        stat_valid_loss.append(valid_loss)
        if ((2 * tp + fp + fn) > 0):
            V_f1 = (2 * tp) / (2 * tp + fp + fn)
        stat_valid_tp.append(tp)
        stat_valid_fp.append(fp)
        stat_valid_fn.append(fn)
        stat_valid_f1_score.append(V_f1)

        # _______________________Accuracy calculation_____________________
        pointer = 0
        test_accuracy = 0
        f1 = 0
        tp = 0
        fp = 0
        fn = 0
        while True:

            # Calculating the correct batch size.
            if (pointer + batch_sz > test_set_size):
                batch = test_set_size % batch_sz
            else:
                batch = batch_sz

            # Training on batch.
            # summary, acc = sess.run([merged, corrects], feed_dict={X: X_test[pointer:pointer + batch], \
            #                                                        Y: Y_test[pointer:pointer + batch]})
            acc = sess.run( corrects, feed_dict={X: X_test[pointer:pointer + batch], \
                                                 Y: Y_test[pointer:pointer + batch]})

            # Calculating the F1 score.
            pred = sess.run(pred_cls,
                            feed_dict={X: X_test[pointer:pointer + batch], Y: Y_test[pointer:pointer + batch]})
            true = np.argmax(Y_test[pointer:pointer + batch], axis=1)
            for i in range(batch):
                if(true[i]==0):
                    if(pred[i]==0):
                        tp+=1
                    else:
                        fn+=1
                else:
                    if(pred[i]==0):
                        fp+=1

            test_accuracy += acc
            pointer += batch
            #test_writer.add_summary(summary, epoch * pointer)

            # Epoch termination condition.
            if (pointer == test_set_size):
                break
        accuracy = test_accuracy/test_set_size
        if ((2 * tp + fp + fn) > 0):
            f1 = (2 * tp) / (2 * tp + fp + fn)
        stat_test_accuracy.append(accuracy)
        stat_test_tp.append(tp)
        stat_test_fp.append(fp)
        stat_test_fn.append(fn)
        stat_test_f1_score.append(f1)

        if (epoch-1 != len(stat_train_loss)):
            file = "C:\\Users\\Username\\Desktop\\Epilepsy\\logs\\graphing\\Patient_"+Patient+"\\log_"+str('log10_')+str('priors')+".csv"
            log = open(file, 'a')
            for i in range(epoch-1, len(stat_train_loss)):
                log.write( str(stat_train_loss[i]) + "," + str(stat_valid_loss[i]) + "," + \
                           str(stat_valid_tp[i]) + "," + str(stat_valid_fp[i]) + "," + str(stat_valid_fn[i]) + "," + \
                           str(stat_valid_f1_score[i]) + "," + str(stat_test_accuracy[i]) + "," + str(stat_test_tp[i]) + "," + \
                           str(stat_test_fp[i]) + "," + str(stat_test_fn[i]) + "," + str(stat_test_f1_score[i]) )
                log.write("\n")
            log.close()

        ##_______________________________Early Stopping______________________________
        my_metric = 0.5*valid_loss + 0.5*(1-V_f1);

        if my_metric < min_my_metric:
            print('Best epoch = ' + str(epoch))
            min_my_metric = my_metric;
            best_epoch = epoch
            early_stop_counter = 0
            save_variables(sess, saver, epoch, Patient)
            best_epoch = epoch
            best_f1_score = f1
            best_accuracy = accuracy
        else:
            early_stop_counter += 1

        ##______________________________Print Epoch Info____________________________
        print('Epoch : ', epoch)
        print('Early Stopping : ' + str(early_stop_counter) + "/" + str(n_early_stop_epochs))
        print('Learning Rate : ', learning_rate)
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

        if early_stop_counter > n_early_stop_epochs :
            # too many consecutive epochs without surpassing the best model
            print('stopping early')
            break

print('Best epoch : ', best_epoch)
print('Best accuracy : ', best_accuracy)
print('Best F1-score : ', best_f1_score)