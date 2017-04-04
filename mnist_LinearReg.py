'''
implemented by Jamal Alikhani for learning purposes, Feb 2017
keywords: Tensoreflow, linear logistic regression, MNIST

'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
x_batch, y_batch_onehot = mnist.train.next_batch(5)
print(y_batch_onehot)

mnist.test.cls = np.array([lable.argmax() for lable in mnist.test.labels])

def plot_mnist_images(images, cls_true, cls_pred=None, img_name="image_out"):
	
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
	plt.savefig(img_name+'.png')
	plt.show()
	

#plot_mnist_images(mnist.test.images[10:19],mnist.test.cls[10:19])

def plot_weights_images(w,ttl=None, img_name="weight_out"):
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
	plt.savefig(img_name+'.png')
	plt.show()

def plot_confusion_matrix(cls_true, cls_pred):  
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
   
    print("confusion matrix: ")
    print(cm)    
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    num_classes = 10
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')  
    plt.savefig("mnist_LinearReg_confusion_matrix"+'.png')  
    plt.show()
	
# ............ TensorFlow Graph Constructio ........................
# Placeholder Variables:
feature_size = 28*28
class_size = 10
x = tf.placeholder(tf.float32, [None,feature_size])
y_true_onehot = tf.placeholder(tf.float32,[None,class_size])  #one hote encoded true lables
y_true_int = tf.placeholder(tf.int64,[None])

# Model Variables:
weights = tf.Variable(tf.zeros([feature_size,class_size]))
biases = tf.Variable(tf.zeros([class_size]))

# Model: logistic regression
logits = tf.matmul(x, weights) + biases
y_pred_onehot = tf.nn.softmax(logits)
y_pred_int = tf.argmax(y_pred_onehot,dimension=1)

# Cost
cost_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true_onehot)
cost = tf.reduce_mean(cost_entropy)

# Optimization
Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

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
	plot_mnist_images(wrong_images[:9],wrong_true_cls[:9],wrong_pred_cls[:9],img_name="mnist_linearreg_wrongpred")
	cls_true = mnist.test.cls	
	plot_confusion_matrix(cls_true, cls_pred)

	# weights show
	w = sess.run(weights)
	plot_weights_images(w,img_name="mnist_linearreg_weights")



