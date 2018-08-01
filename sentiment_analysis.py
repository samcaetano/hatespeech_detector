# <script-name>.py <optimizer> <dataset> <embed_method>
from __future__ import print_function

import tensorflow as tf
from loader import create_feature_sets_and_labels
from sklearn.model_selection import StratifiedKFold
import numpy as np
import timeit
import sys
import re

batch_size = 50
hm_epochs = 50
dataset = ''
optimizer = ''
filter_size = [3, 4, 5]
num_filters = 100
vocab_size = 0
embedding_size = 0
opt = ''

path_set = sys.argv[2]

if path_set == 'offcombr3':
	dataset = 'datasets/OffComBR3.arff'
elif path_set == 'offcombr2':
	dataset = 'datasets/OffComBR2.arff'
elif path_set == 'hatespeech':
	dataset = 'datasets/hatespeech_ptbr.arff'

opt = sys.argv[1]
embed_method = sys.argv[3]

train_x, train_y, test_x, test_y, vocabulary, embed = create_feature_sets_and_labels(dataset, embed_method)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_y_modified = []
for sample in train_y:
	if sample[0] == 1:
		train_y_modified.append(0)
	else:
		train_y_modified.append(1)

sequence_length = train_x.shape[1]
num_classes = train_y.shape[1]


x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, num_classes])


if embed_method == 'none':
	vocab_size = len(vocabulary) # 407
	embedding_size = 128
else:
	vocab_size = embed.shape[0]
	embedding_size = embed.shape[1]

seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

print('Using dataset ', dataset)
print('train set ', train_x.shape, train_y.shape)
print('test set ', test_x.shape, test_y.shape)

print('Hyperparameters:')
print('sequence_length: {}'.format(sequence_length))
print('num_classes: {}'.format(num_classes))
print('vocab_size: {}'.format(vocab_size))
print('embedding_size: {}'.format(embedding_size))


def train_neural_net():
	k = 0
	cross_validation_scores = []
	f1_scores = []

	for train, eval in kfold.split(train_x, train_y_modified):
		print(k+1,'-fold.')
		prediction, keep_prob  = convolutional_network_model()

		with tf.name_scope('loss'):
			cost = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y)
		cost = tf.reduce_mean(cost)


		if opt == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
		elif opt == 'rmsprop':
			optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)
		elif opt == 'adagrad':
			optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(cost)
		elif opt == 'adadelta':
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cost)

		saver = tf.train.Saver()                

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(hm_epochs):
				epoch_loss = 0
				i = 0
				while i < len(train_x[train]):#for _ in range(int(mnist.train.num_examples/batch_size)):
					start = i
					end = i + batch_size
					batch_x = np.array(train_x[train][start:end])
					batch_y = np.array(train_y[train][start:end])
							
					_, c = sess.run([optimizer, cost],
									feed_dict={x: batch_x,
											   y: batch_y,
											   keep_prob: 0.5})
					epoch_loss += c
					i += batch_size
				print('epoch ', epoch+1, ' completed of ', hm_epochs, '. Loss: ', epoch_loss),

				with tf.name_scope('val_accuracy'):
					correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
					accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				print('val accuracy:', accuracy.eval({x: train_x[eval], y: train_y[eval], keep_prob: 1.0}))

			with tf.name_scope('test_accuracy'):
				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				test_acc = accuracy.eval({x: test_x, y: test_y, keep_prob: 1.0})
			
			cross_validation_scores.append(test_acc)

			with tf.name_scope('f1_score'):
				argmax_prediction = tf.argmax(prediction, 1)
				argmax_y = tf.argmax(test_y, 1)

				TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
				TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
				FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
				FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)	

				recall = TP/(TP+FN)
				precision = TP/(TP+FP)

				f1 = (2*recall * precision) / (recall + precision)
			f1_score = f1.eval({x: test_x, y:test_y, keep_prob: 1.0})

			f1_scores.append(f1_score)

			k+=1

			print('Test acc: ', test_acc, ' f1_score: ', f1_score)
			save_path = saver.save(sess, 'saved_models/'+opt+'.'+path_set+'.'+embed_method)
			print('Saving model to ', save_path)				

			print()

	with open('training_results/'+opt+'_'+sys.argv[2]+'_'+embed_method, 'w') as f:
		f.write(str(cross_validation_scores))
		f.write(str(f1_scores))

		print('Average test accuracy: ', np.mean(cross_validation_scores))
		print('Average F1 scores', np.mean(f1_scores))

		f.close()

def convolutional_network_model():
	with tf.name_scope('embedding'), tf.device('/cpu:0'):

		W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))

		if embed_method != 'none':
			embedding_init = W.assign(embedding_placeholder)
			sess = tf.Session()
			sess.run(embedding_init, feed_dict={embedding_placeholder: embed})
	
		embedded_chars = tf.nn.embedding_lookup(W, x) # return format [None, sequence_length, embedding_size]
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
		#print('embedding: ', embedded_chars_expanded.get_shape())

	with tf.name_scope('conv1'):
		W = tf.Variable(tf.truncated_normal([filter_size[0], embedding_size, 1, num_filters], stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

		conv_1 = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1],
			padding='VALID')

		conv_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b))
		#print('conv1: ', conv_1.get_shape())

	with tf.name_scope('conv2'):
		W = tf.Variable(tf.truncated_normal([filter_size[1], embedding_size, 1, num_filters], stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

		conv_2 = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1],
			padding='VALID')

		conv_2 = tf.nn.relu(tf.nn.bias_add(conv_2, b))
		#print('conv2: ', conv_2.get_shape())

	with tf.name_scope('conv3'):
		W = tf.Variable(tf.truncated_normal([filter_size[2], embedding_size, 1, num_filters], stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

		conv_3 = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1],
			padding='VALID')

		conv_3 = tf.nn.relu(tf.nn.bias_add(conv_3, b))
        #        print('conv3: ', conv_3.get_shape())

	with tf.name_scope('pool1'):
		h_pool_1 = tf.nn.max_pool(conv_1,
			ksize=[1, sequence_length - filter_size[0] + 1, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')
        #        print('pool1: ', h_pool_1.get_shape())

	with tf.name_scope('pool2'):
		h_pool_2 = tf.nn.max_pool(conv_2,
			ksize=[1, sequence_length - filter_size[1] + 1, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')
        #        print('pool2: ', h_pool_2.get_shape())

	with tf.name_scope('pool3'):
		h_pool_3 = tf.nn.max_pool(conv_3,
			ksize=[1, sequence_length - filter_size[2] + 1, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')
        #        print('pool3: ', h_pool_3.get_shape())

	h_pool = tf.concat([h_pool_1, h_pool_2, h_pool_3], 1)
    #    print('h_pool_concat: ', h_pool.get_shape())

	num_filters_total = len(filter_size) * num_filters
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    #    print('h_pool_flat: ', h_pool_flat.get_shape())

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_dropout = tf.nn.dropout(h_pool_flat, keep_prob)
    #            print('h_dropout: ', h_dropout.get_shape())

	with tf.name_scope('output'):
		W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]))

		scores = tf.add(tf.matmul(h_dropout, W), b)

	return scores, keep_prob

s = timeit.default_timer()
#train_neural_net()
e = timeit.default_timer()
print('Total time {}s'.format(e-s))
