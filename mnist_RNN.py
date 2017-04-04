'''
implemented by Jamal Alikhani for learning purposes, Feb 2017
keywords: Tensoreflow, RNN, MNIST

'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
x_batch, y_batch_onehot = mnist.train.next_batch(5)

mnist.test.cls = np.array([lable.argmax() for lable in mnist.test.labels])

def plot_mnist_images(images, cls_true, cls_pred=None):
	
	#fig.subplot_adjust(hspace=0.3, wspace=0.3)
	for i in range(9):
		img = np.array(images[i]).reshape((28,28))
		ax = plt.subplot(3,3,i+1)
		ax.imshow(img,cmap='binary')
		ax.set_xticks([])
		ax.set_yticks([])
		if cls_pred is None:
			ax.set_xlabel("True = {0}".format(cls_true[i]))
		else:
			ax.set_xlabel("True = {0}, Pred = {1}".format(cls_true[i], cls_pred[i]))
	plt.show()

#plot_mnist_images(mnist.test.images[10:19],mnist.test.cls[10:19])

def plot_weights_images(w,ttl=None):
	w = np.array(w)
	wmin = np.min(w)
	wmax = np.max(w)
	
	for i in range(10):
		img = np.array(w[:,i]).reshape((28,28))	
		ax = plt.subplot(4,3,i+1)	
		ax.imshow(img,vmin=wmin, vmax=wmax, cmap='seismic')
		ax.set_xlabel("weights: {0}".format(i))
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()
	
# ............ TensorFlow Graph Constructio ........................
# Placeholder Variables:
feature_size = 28*28
chunk_size = 28
num_chunks = int(feature_size/chunk_size)
class_size = 10
rnn_layer_size = 128

# place holder
x = tf.placeholder(tf.float32, [None, feature_size])
x = tf.reshape(x,[-1, num_chunks, chunk_size])
x = tf.transpose(x,[1,0,2])
x = tf.reshape(x,[-1, chunk_size])
x = tf.split(0, num_chunks, x)

y_true_onehot = tf.placeholder(tf.float32,[None,class_size])  #one hote encoded true lables
y_true_int = tf.placeholder(tf.int64,[None])

# Model Variables:
weights = tf.Variable(tf.zeros([rnn_layer_size,class_size]))
biases = tf.Variable(tf.zeros([class_size]))

# LSTM cell:
lstm_cell = rnn_cell.BasicLSTMCell(rnn_layer_size)
lstm_out, _ = rnn.rnn(lstm_cell, x, dtype=tf.float32)

# Model: logistic regression
logits = tf.matmul(lstm_out, weights) + biases
y_pred_onehot = tf.nn.softmax(logits)
y_pred_int = tf.argmax(y_pred_onehot,dimension=1)

# Cost
cost_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true_onehot)
cost = tf.reduce_mean(cost_entropy)

# Optimization
Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Model Performance
Correct_prediction = tf.equal(y_pred_int,y_true_int)
accuracy = tf.reduce_mean(tf.cast(Correct_prediction, tf.float64))

# ......................... Running The TF Graph ....................
# Open a session
with tf.Session() as sess:
	# initialization
	sess.run(tf.initialize_all_variables())

	#training:
	batch_size = 100
	num_iteration = int(mnist.train.num_examples/batch_size)
	for i in range(num_iteration):
		x_batch, y_batch_onehot = mnist.train.next_batch(batch_size)
		feed_dict_train = {x:x_batch, y_true_onehot:y_batch_onehot}
		sess.run(Optimizer,feed_dict=feed_dict_train)
		w = sess.run(weights)

	# Accuracy
	feed_dict_test = {x:mnist.test.images, y_true_onehot:mnist.test.labels, y_true_int:mnist.test.cls}
	acc = sess.run(accuracy, feed_dict = feed_dict_test)
	print("Accuracy on test-set: {0:.1%}".format(acc))
	correct, cls_pred = sess.run([Correct_prediction, y_pred_int], feed_dict = feed_dict_test)
	incorrect = (correct==False)
	wrong_images = mnist.test.images[incorrect]
	wrong_pred_cls = cls_pred[incorrect]
	wrong_true_cls = mnist.test.cls[incorrect]
	plot_mnist_images(wrong_images[:9],wrong_true_cls[:9],wrong_pred_cls[:9])

	# weights show
	w = sess.run(weights)
	plot_weights_images(w)



