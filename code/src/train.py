import os
import numpy as np
import tensorflow as tf
from model import CNN
from lib.model_io import get_modle_id
import datetime

seed = int(datetime.datetime.utcnow().strftime('%m%d%H%M%S'))

__author__ = "Konidaris Vissarion, Logothetis Fragoulis"

Patient = "8"
model_id = get_modle_id(Patient, seed)

# Create the network
network = CNN(patient=Patient, model_id=model_id, seed=seed)

# Train the network
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#with tf.Session() as sess:

	try:
		network.fit(sess)
	except KeyboardInterrupt:
		print("Why you terminate da learning? :(")
		sess.close()