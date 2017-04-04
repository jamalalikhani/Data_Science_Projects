'''
implemented by Jamal Alikhani for learning purposes, Feb 2017
keywords: Tensoreflow, CNN (2 conv layers + 1 fully connected), MNIST

'''
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import math
from sklearn.metrics import confusion_matrix

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)
mnist.test.cls = np.array([lable.argmax() for lable in mnist.test.labels])

print(" Size of:")
print(" - Train Set:\t\t{0}".format(len(mnist.train.labels)))
print(" - Test Set:\t\t{0}".format(len(mnist.test.labels)))
print(" - Validation Set:\t{0}".format(len(mnist.validation.labels)))

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
	plt.savefig(img_name + '.png')
	plt.show()

def plot_flat_weights(w, filter_size, num_filters, img_name="weight_out"):
	wmin = np.min(w)
	wmax = np.max(w)
	a = int(math.sqrt(num_filters))
	filter_length = filter_size**2	
	plt.figure(figsize=(15,15))
	for i in range(num_filters):
		n1 = i*filter_length
		n2 = (i+1)*filter_length		
		img = np.array(w[n1:n2]).reshape((filter_size,filter_size))	
		ax = plt.subplot(a,a,i+1)	
		ax.imshow(img,vmin=wmin, vmax=wmax, interpolation='nearest', cmap='seismic')
		ax.set_xlabel("Filter: {0}".format(i+1))
		ax.set_xticks([])
		ax.set_yticks([])
	plt.savefig(img_name+'.png')
	plt.show()

def plot_flat_images(img, filter_size, num_filters, img_name="flat_image_out"):	
	a = int(math.sqrt(num_filters))
	filter_length = filter_size**2	
	plt.figure(figsize=(15,15))
	for i in range(num_filters):
		img1 = np.zeros((filter_size,filter_size))
		for k in range(filter_size):
			for j in range(filter_size):
				img1[k][j] = img[0][k][j][i]			
		ax = plt.subplot(a,a,i+1)	
		ax.imshow(img1,cmap='binary')
		ax.set_xlabel("Image Out: {0}".format(i+1))
		ax.set_xticks([])
		ax.set_yticks([])
	plt.savefig(img_name+'.png')
	plt.show()

def plot_confusion_matrix(cls_true, cls_pred):      

    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
   
    print(cm)    
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')  
    plt.savefig("mnist_CNN_confusion_matrix"+'.png')  
    plt.show()

# ......................... CNN ...................
# CNN configuration:
# Conv layer I:
filter_size1 = 5
num_filter1 = 16

# Conv layer II:
filter_size2 = 5
num_filter2 = 36

#fully connected layer:
fc_size = 128 

img_size = 28
img_size_flat = 28*28
img_shape=(img_size, img_size)
num_channels = 1 # 1 channel for gray scale (3 channles for RGB)
num_classes = 10

# placeholder variables:
x2D = tf.placeholder(tf.float32, [None, img_size_flat], name='x2D')
x4D = tf.reshape(x2D, [-1, img_size, img_size, num_channels]) # -1 can also be used to infer the shape
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.argmax(y_true, dimension=1)

# ~~~~~~~~~~~ Layers: ~~~~~~~~~~~~~~

def new_weights(shape): #4D
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):  #1D
	return tf.Variable(tf.constant(0.05, shape=[length]))

# Convolutional layer: a 4D tensor
def conv_layer(input,  #prev layer
	num_input_channels,  # num of channels in prev layer
	filter_size,
	num_filters,
	use_max_pooling=True): # use 2x2 max-pooling

	shape = [filter_size, filter_size, num_input_channels, num_filters]

	weights = new_weights(shape=shape)
	biases = new_biases(length=num_filters)

	layer = tf.nn.conv2d(input=input,
		filter=weights,
		strides=[1,1,1,1],  #strides=[img-num,x-axis moving, y-axis moving, input-channel]
		padding='SAME')

	layer += biases

	if use_max_pooling:  # max-pooling occures after convolution
		layer = tf.nn.max_pool(value=layer,
			ksize=[1,2,2,1],  #max-pool size I think!
			strides=[1,2,2,1],
			padding='SAME')

	# rectified linear unit (ReLU): x = max(0,x)
	layer = tf.nn.relu(layer)

	return layer, weights

def tensor4D_to_tensor2D(layer4D):
	# 4Dlayer = [num_images, img_width, img_height, num_channels]
	layer4D_dim = layer4D.get_shape()
	
	# 2D layer = [num_images, num_features] & num_features=img_width*img_height*num_channels
	num_features = np.array(layer4D_dim[1:4],dtype=int).prod()

	layer2D = tf.reshape(layer4D,[-1,num_features]) 

	return layer2D, num_features

def fullyConnect_layer(input,
	num_inputs,
	num_outputs,
	use_relu=True):
	
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	layer = tf.matmul(input,weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer, weights

# ============= Layer Construction:

# 1st conv layer: 1 img in, 16 filters
layer_conv1, weights_conv1 = conv_layer(input=x4D,num_input_channels=num_channels,  
	filter_size=filter_size1,	num_filters=num_filter1,	use_max_pooling=True) 

layer_conv2, weights_conv2 = conv_layer(input=layer_conv1,num_input_channels=num_filter1,  
	filter_size=filter_size2, num_filters=num_filter2, use_max_pooling=True) 


layer_conv2_2D, num_features_fc = tensor4D_to_tensor2D(layer_conv2)


layer_fc, weights_fc = fullyConnect_layer(input=layer_conv2_2D,	
	num_inputs=num_features_fc,	num_outputs=fc_size, use_relu=True)

layer_output, weights_output = fullyConnect_layer(input=layer_fc,	
	num_inputs=fc_size,	num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_output)  #brings all the outputs in the rangeof 0 and 1

y_pred_cls = tf.argmax(y_pred, dimension=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Training
sess = tf.Session()
sess.run(tf.initialize_all_variables())
        
batch_size = 50
print("Batch size is {0}: ".format(batch_size))
max_iter = 1000
num_iter = min(max_iter,int(len(mnist.train.labels)/batch_size))
for n in range(max_iter):
	x_train, y_train = mnist.train.next_batch(batch_size)
	feed_dict_train = {x2D: x_train, y_true:y_train}
	sess.run(optimizer, feed_dict=feed_dict_train)
	if n%100==0:
		acc = sess.run(accuracy,feed_dict=feed_dict_train)
		print("At iter {0} the accuracy is: {1}".format(n+1,acc))

#accuracy:
do_accuracy = False
if do_accuracy:	
	feed_dict_test = {x2D: mnist.test.images[:100, :], y_true: mnist.test.labels[:100, :]}
	acc = sess.run(accuracy, feed_dict=feed_dict_test)
	print("Accuracy: 0:.1%".format(acc))

	correct, y_pred_cls = sess.run([correct, y_pred_cls], feed_dict=feed_dict_test)
	incorrect = (correct==False)
	wrong_images = mnist.test.images[incorrect]
	wrong_pred_lables = y_pred_cls[incorrect]
	wrong_true_lables = mnist.test.cls[incorrect]
	plot_mnist_images(wrong_images, cls_true=wrong_true_lables, cls_pred=wrong_pred_lables, img_name="mnist_CNN_wrongpred")
	cls_true = mnist.test.cls
	plot_confusion_matrix(cls_true[:100], y_pred_cls)

# plot the filters for each layer:

feed_dict_single = {x2D:[mnist.test.images[0]]}
imge_layer_conv1, weights1, imge_layer_conv2, weights2 = sess.run([layer_conv1, weights_conv1, layer_conv2, weights_conv2],feed_dict=feed_dict_single)

w1 = sess.run(tf.reshape(weights1,[filter_size1*filter_size1*num_filter1]))
w2 = sess.run(tf.reshape(weights2,[-1,filter_size1*filter_size1*num_filter2]))

plot_flat_weights(w1, filter_size=filter_size1, num_filters=num_filter1, img_name="mnist_CNN_layer1_filter_weight")
plot_flat_images(imge_layer_conv1, filter_size=14, num_filters=num_filter1, img_name="mnist_CNN_layer1out_image")
plot_flat_images(imge_layer_conv2, filter_size=7, num_filters=num_filter2, img_name="mnist_CNN_layer2out_image")

sess.close()



