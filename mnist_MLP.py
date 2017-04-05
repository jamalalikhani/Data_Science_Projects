'''
implemented by Jamal Alikhani for learning purposes, Feb 2017
keywords: Tensoreflow, MLP, MNIST

'''

'''
input > weight > hidden layer 1 (activation function) >
weights > hidden layer 2(activation function) >
weights > output layer

compare output to intended output > cost function (cross entropy)
optimizer > minimize cost (AdamOptmizer...SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

mnist.test.cls = np.array([lable.argmax() for lable in mnist.test.labels])

# 10 classes, 0 to 9 
'''
one_hot=True
0 = [1 0 0 0 0 0 0 0 0 0 0]
1 = [0 1 0 0 0 0 0 0 0 0 0]
2 = [0 0 1 0 0 0 0 0 0 0 0]
3 = [0 0 0 1 0 0 0 0 0 0 0]

'''
n_nodes_hl1 = 784
n_nodes_hl2 = 128
n_nodes_hl3 = 128

n_classes = 10
batch_size = 100

# image size = 28x28 = 784 
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# input_data * weights + biases
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
	                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
	                  'biases':tf.Variable(tf.random_normal([n_classes]))}                                                      

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)  # activation function


	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)  # activation function


	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)  # activation function

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	hm_epochs = 100

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				ex, ey = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: ex, y: ey})
				epoch_loss += c
			print('Epoch ', epoch, ' completed out of ', hm_epochs,' loss: ',epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))	
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy:",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)



