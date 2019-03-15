import os
import tensorflow as tf
from model import CNN
from lib.model_io import get_modle_id

__author__ = "Konidaris Vissarion, Logothetis Fragoulis"

# Type the model of a specific patient
Patient = "8"
model_id = "1"
seed = 628230015

# Create the network
network = CNN(patient=Patient, model_id=model_id, seed=seed, train=False)

# Recover the parameters of the model
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#with tf.Session() as sess:

    network.init_Variables(sess)
    network.restore_variables(sess)

    # Iterate through test files and calculate the classification scores
    accuracy, tp, fp, fn, f1 = network.test_Score(sess)
    print('')
    print('####################################')
    print('Accuracy : ', accuracy)
    print('TP : ', tp)
    print('FP : ', fp)
    print('FN : ', fn)
    print('F1-score : ', f1)
    print('####################################')
    print('')

    sess.close()


