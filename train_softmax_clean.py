#!/usr/bin env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import argparse
from utils import normalize, decode
from model import graph

def input_pipeline(trainpath, validationpath, buffer_size, batch_size):

	# Skip the header and filter any comments.
	

	training_dataset = tf.data.TextLineDataset(trainpath).skip(1).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))
	validation_dataset = tf.data.TextLineDataset(validationpath).skip(1).filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))

	default_values = [[0.0] for _ in range(785)]

	# The dataset api reads the csv as text.
    # Using the below function we can split the text into labels and pixels.
	training_dataset = (training_dataset.cache().map(lambda x: decode(x, default_values)))
	validation_dataset = (validation_dataset.cache().map(lambda x: decode(x, default_values)))

	# Normalize the dataset to 0-1 range
	training_dataset = training_dataset.map(lambda label, pixel: tf.py_func(normalize, [label, pixel], [tf.float32, tf.float32]))
	validation_dataset = validation_dataset.map(lambda label, pixel: tf.py_func(normalize, [label, pixel], [tf.float32, tf.float32]))
	
	# Randomly shuffles the dataset
	training_dataset = training_dataset.shuffle(buffer_size=buffer_size)

	# Creating batchs here for training
	training_dataset = training_dataset.batch(batch_size)
	validation_dataset = validation_dataset.batch(batch_size)
	
	# A feedable iterator is defined by a handle placeholder and its structure. We
	# could use the `output_types` and `output_shapes` properties of either
	# `training_dataset` or `validation_dataset` here, because they have
	# identical structure.
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
	next_element = iterator.get_next()
	
	# You can use feedable iterators with a variety of different kinds of iterator
	# (such as one-shot and initializable iterators).
	training_iterator = training_dataset.make_initializable_iterator()
	validation_iterator = validation_dataset.make_initializable_iterator()
	
	return next_element, handle, training_iterator, validation_iterator

def train(batch_size, learning_rate, x, y):
	logits = graph(x)
	_y = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits), axis=0)
	acc = tf.equal(tf.argmax(logits, 1), tf.argmax(_y, 1))
	acc = tf.reduce_mean(tf.cast(acc, tf.float32))
	global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
	return loss, acc, train_op, global_step

def print_results(iteration, losses, accuracy):
    print("Batch: {0:5d} loss: {1:0.3f} accuracy: {2:0.3f}"
          .format(iteration, np.mean(losses), np.mean(accuracy)))

def print_results_epoch(iteration, losses, accuracy):
    print("Epoch: {0:5d} loss: {1:0.3f} accuracy {2:0.3f}"
          .format(iteration+1, np.mean(losses), np.mean(accuracy)))

def print_results_val(losses, accuracy):
    print("Validation loss: {0:0.3f} accuracy {1:0.3f}"
          .format(np.mean(losses), np.mean(accuracy)))

# def print_accuracy(iteration, acc):
# 	print("Batch: {0:5d} Accuracy: {1:0.3f}"
#           .format(iteration, np.mean(acc)))

# def print_accuracy_epoch(iteration, acc):
#     print("Epoch: {0:5d} Accuracy: {1:0.3f}"
#           .format(iteration+1, np.mean(acc)))

def parser(argv):
	parser = argparse.ArgumentParser(description='Trains a Deep Neural Network on Fashion MNIST Data')
	parser.add_argument('--train_csv', default='training.csv', type=str, required=True, help='Path to the training csv.')
	parser.add_argument('--validation_csv', default='validation.csv', type=str, help='Path to the validation csv.')
	parser.add_argument('--batch_size', default=100, type=int, help='Batch Size of one iteration.')
	parser.add_argument('--buffer_size', default=10000, type=int, help='Buffer Size for random selection of images.')
	parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate.')
	parser.add_argument('--nrof_epochs', default=20, type=int, help='Number of Epochs for training.')
	parser.add_argument('--log_dir', default='./log', type=str, help='Location of log.')
	parser.add_argument('--model_dir', default='./model', type=str, help='Location of saved model.')
	args = parser.parse_args()
	return args

def main(args):
	trainpath = args.train_csv
	validationpath = args.validation_csv
	batch_size = args.batch_size
	buffer_size = args.buffer_size
	learning_rate = args.lr
	nepochs = args.nrof_epochs
	logdir = args.log_dir
	savepath = args.model_dir

	if not os.path.exists(trainpath):
		raise IOError('Training file does not exist')

	if not buffer_size or not batch_size:
		raise ValueError('Please provide valid value for buffer_size and batch_size')

	if not os.path.exists(savepath):
		os.makedirs(savepath)

	x = tf.placeholder('float32',shape=[batch_size,None])
	y = tf.placeholder('int32',shape=[batch_size])

	next_element, handle, training_iterator, validation_iterator = input_pipeline(trainpath, validationpath, buffer_size, batch_size)
	loss, acc, train_op, global_step = train(batch_size, learning_rate, x, y)

	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', acc)
	merged = tf.summary.merge_all()

	training_loss = []
	epoch_loss = []
	train_acc = []
	epoch_acc = []
	val_loss = []
	val_acc = []

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.group(tf.global_variables_initializer(), 
	                      tf.local_variables_initializer()))

		if os.path.exists(os.path.join(savepath,"checkpoint")):
			print("="*30)
			print("Restoring existing model..")
			print("="*30)
			print()
			saver.restore(sess, os.path.join(savepath, "model.ckpt"))

		train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
		# train_writer.add_graph(tf.get_default_graph())
		# The `Iterator.string_handle()` method returns a tensor that can be evaluated
	    # and used to feed the `handle` placeholder.

		training_handle = sess.run(training_iterator.string_handle())
		validation_handle = sess.run(validation_iterator.string_handle())

		for i in range(nepochs):
			sess.run(training_iterator.initializer)
			while True:
				try:
					label_batch, image_batch = sess.run(next_element, feed_dict={handle: training_handle})
					summary, _loss, _acc, _, g = sess.run([merged, loss, acc, train_op, global_step], feed_dict = {x:image_batch, y:label_batch})
					training_loss.append(_loss)
					epoch_loss.append(_loss)
					train_acc.append(_acc)
					epoch_acc.append(_acc)
					if tf.train.global_step(sess, global_step)%10==0:
						train_writer.add_summary(summary, g)
						print_results(g, training_loss, train_acc)
						training_loss = []
						train_acc = []
				except tf.errors.OutOfRangeError:
					print('='*60)
					print('Epoch {} Finished !'.format(i+1))
					print_results_epoch(i, epoch_loss, epoch_acc)
					print('='*60)
					print()
					print('Running forward pass on validation set..')
					sess.run(validation_iterator.initializer)
					
					while True:
						try:
							val_label_batch, val_image_batch = sess.run(next_element, feed_dict={handle: validation_handle})
							_val_loss, _val_acc = sess.run([loss, acc], feed_dict = {x:val_image_batch, y:val_label_batch})
							val_loss.append(_val_loss)
							val_acc.append(_val_acc)
						except tf.errors.OutOfRangeError:
							break
					print('='*60)
					print_results_val(val_loss, val_acc)
					print('='*60)
					print()
					break
			# print_results_epoch(i, epoch_loss, epoch_acc)
			epoch_loss = []
			epoch_acc = []
			if int(nepochs - i) <= 2:
				saver.save(sess, os.path.join(savepath,"model.ckpt"))
				print("Model saved in %s" % (savepath))
				print()
	return

if __name__ == '__main__':
	main(parser(sys.argv[1:]))
