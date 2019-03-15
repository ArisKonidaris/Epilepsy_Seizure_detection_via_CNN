from __future__ import division, print_function

import os
import argparse
import json
import librosa
import numpy as np
import tensorflow as tf


def read_model_id(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as fid:
            model_id = int(fid.read())
        fid.close()
    else:
        write_model_id(filename, 1)
        model_id = 0

    return model_id


def write_model_id(filename, model_id):
    model_id_txt = str(model_id)
    with open(filename, 'w') as fid:
        fid.write(model_id_txt)
    fid.close()

def write_seed(Patient, seed):
    model_seed_filename = "C:\\Users\Username\\Desktop\\Epilepsy\\code\\saved_models\\DeepParams\\Patient_"
    model_seed_filename += str(Patient) + "\\Model_Seed"

    with open(model_seed_filename, 'a') as fid:
        fid.write(str(seed)+"\n")
    fid.close()

def get_seed(Patient, model_id):
    model_seed_filename = "C:\\Users\Username\\Desktop\\Epilepsy\\code\\saved_models\\DeepParams\\Patient_"
    model_seed_filename += str(Patient) + "\\Model_Seed"
    with open(model_seed_filename, 'r') as fid:
        seed = fid.readlines()
    fid.close()

    return int(seed[int(model_id)-1])

def get_modle_id(Patient, seed):
    model_id_filename = "C:\\Users\Username\\Desktop\\Epilepsy\\code\\saved_models\\DeepParams\\Patient_"
    model_id_filename += str(Patient) + "\\M_IDs"
    model_id = read_model_id(
        model_id_filename) + 1  # Reserve the next model_id. If file does not exists then create it 
    write_model_id(model_id_filename, model_id)
    write_seed(Patient, seed)

    return model_id
