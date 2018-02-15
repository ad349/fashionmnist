#!/usr/bin env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import argparse
from utils import splitter, normalize
from model import graph

def parser(argv):
    parser = argparse.ArgumentParser(description='Trains a Deep Neural Network on Fashion MNIST Data')
    parser.add_argument('--train_csv', default='training.csv', type=str, required=True, help='Path to the training csv.')
    parser.add_argument('--validation_csv', default='validation.csv', type=str, help='Path to the validation csv.')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch Size of one iteration.')
    parser.add_argument('--buffer_size', default=10000, type=int, help='Buffer Size for random selection of images.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate.')
    parser.add_argument('--nrof_epochs', default=20, type=int, help='Number of Epochs for training.')
    args = parser.parse_args()
    return args

def main():
    # Session creation
    # Sesssion run
    # Epoch counter
    return True

def train():
    # Returns training op
    # Includes optimizer
    return True

def get_loss():
    # Returns softmax loss
    return True

def get_acc(logits, labels):
    # Returns accuracy stats
    pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
    correct_
    return True
