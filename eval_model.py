from __future__ import print_function
import tensorflow as tf
import numpy as np
from loader import create_feature_sets_and_labels

# Off2 - 551
# Off3 - 478
# hsd -  826

dataset='hatespeech_ptbr.arff'
embed_method='none'

train_x, train_y, test_x, test_y, vocabulary, embed = create_feature_sets_and_labels('datasets/'+dataset, embed_method)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

X=np.concatenate((train_x, test_x), axis=0) 
Y=np.concatenate((train_y, test_y), axis=0)

print(X.shape, Y.shape)

sequence_length = X.shape[1]
num_classes = Y.shape[1]

x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32, None)

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('saved_models/OFFCOMBR/adam.offcombr3.glove300.meta')
	saver.restore(sess, 'saved_models/OFFCOMBR/adam.offcombr3.glove300')

	graph = tf.get_default_graph()

	accuracy = graph.get_operation_by_name('test_accuracy_9/Mean').outputs[0]
	f1_score = graph.get_operation_by_name('f1_score_9/div_2').outputs[0]
	x = graph.get_operation_by_name('Placeholder_3').outputs[0]
	y = graph.get_operation_by_name('Placeholder_1_1').outputs[0]
	keep_prob = graph.get_operation_by_name('dropout_9/Placeholder').outputs[0]
	
	x_shape=x.get_shape().as_list()
	print('model: {} | input:{}'.format(x_shape[1], sequence_length))
	cols = x_shape[1]-sequence_length

	zeros = np.zeros((X.shape[0], cols), dtype=float)
	X = np.concatenate((X, zeros), axis=1) 

	batch_size=50
	accs=[]
	f_scores=[]
	for _ in range(10):
		i = 0
		acc=[]
		while i < len(X):
			start = i
			end = i + batch_size
			batch_x = np.array(X[start:end])
			batch_y = np.array(Y[start:end])
				
			acc.append(accuracy.eval({x: batch_x, y: batch_y, keep_prob: 1.0}))
			i += batch_size
		accs.append(np.average(acc))
		f_scores.append(f1_score.eval({x: X[:125], y: Y[:125], keep_prob: 1.0}))
	
	print('f1_score: {}'.format(np.average(f_scores)))
	print('average accuracy: {}'.format(np.average(accs)))

	
