# Applying TensorFlow for classifying [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)
In this attempt, I have applied TensorFlow (puthon 3.5 on Ubuntu) to construct the follwing four models:
1. Linear Regression with 1 fully connected layer
2. Multilayer Perceptron Model (MLP) with 3 fully connected layers
3. Recurrent Neural Network (RNN) with 1 NN layer inside an LSTM cell
4. Convolutional Neural Network (CNN) with 2 Convolutional/pooling layers and 1 fully connected layer

*mnist* are a set of images of handwritten English numbers `0 to 9` in gray-scale of `28*28` resolution like:![mnist_sample](https://cloud.githubusercontent.com/assets/22183834/24684604/162534be-195c-11e7-9493-b20f1e764728.png)

The goal was to design a NN with different approaches to train by using 60,000 training data set. 

**Refrences:**
most of the materials are taken from:
* [TensorFlow Example for MNIST](https://www.tensorflow.org/get_started/mnist/pros)
* [Magnus Pedersen's tutorial](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
* [sentdex's tutorial channel](https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

You are welcome to modify these codes and use them in your own projects. Please contact me if you have any questions/comments via: jamal.alikhani@gmail.com.

## TensorFlow Architecture for Neural Network (general perspective)
  
1. **TensorFlow Graph Construction**
   1. Placeholder Variables:  
   ```
      x = tf.placeholder(tf.float32, [None, feature_size])
      y_true = tf.placeholder(tf.float32, [None, class_size]) 
   ```  
   2. Model Variables:  
   ```
      weights = tf.Variable(tf.zeros([feature_size, class_size]))
      biases = tf.Variable(tf.zeros([class_size]))
   ```  
   3. Model (logistic regression for example):  
   ```
      logits = tf.matmul(x, weights) + biases
      y_pred = tf.nn.softmax(logits)
   ```  
   4. Cost funtion:  
   ```
      cost_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true_onehot)
      cost = tf.reduce_mean(cost_entropy)
   ```  
   5. Optimizer algorithm:  
   ```  
      Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
   ```  
   6. Model Performance:
   ```
      Correct_prediction = tf.equal(y_pred, y_true)
      accuracy = tf.reduce_mean(tf.cast(Correct_prediction, tf.float64))
   ```

2. **Running The TF Graph**
   1. Open a session:
   ```
      sess = tf.Session()
   ```  
   2. initialization:
   ```
      sess.run(tf.initialize_all_variables())
   ```  
   3. Training:
   ```
      batch_size = 256
      num_iteration = int(mnist.train.num_examples/batch_size)
      for i in range(num_iteration):
		x_batch, y_batch_onehot = mnist.train.next_batch(batch_size)
		feed_dict_train = {x:x_batch, y_true_onehot:y_batch_onehot}
		sess.run(Optimizer,feed_dict=feed_dict_train)
		w = sess.run(weights)
   ```  
   4. Accuracy evaluation:
   ```
      feed_dict_test = {x:mnist.test.images, y_true_onehot:mnist.test.labels, y_true_int:mnist.test.cls}
      acc = sess.run(accuracy, feed_dict = feed_dict_test)
   ```
   5. weights show
   ```
      weights = sess.run(weights)
   ```
  
   6. Close the session:
   ```
      sess.close()
   ```
   
## Results
#### Linear Regression Model:

A fully connected layer of size `28*28=784` was considered that each neuron was mapped to a particuar pixcel in the image. After training, the weights are illustrated in the image form:
![mnist_linearreg_weights](https://cloud.githubusercontent.com/assets/22183834/24684621/2bc602c6-195c-11e7-8687-7d89e335aecd.png)  

where redish color shows positive weights and blueish shows negetive weights. Interestigly (and expectedly) the weights of each number illustrate the same value on the graph. This is the most advantage of using Linear Regression method that it's weights are are interpretable. 

The matrix confusion is also showing below:
```
 [[ 957    0    2    3    0    8    7    1    2    0]
 [   0 1112    2    3    1    2    4    0   11    0]
 [  13   13  873   23   14    2   20   23   40   11]
 [   5    2   17  889    1   44    6   15   18   13]
 [   1    6    4    0  889    1   13    1    8   59]
 [  10    5    1   31   12  775   17    7   24   10]
 [  17    3    4    2    8   22  897    1    4    0]
 [   3   20   26    4   11    0    0  917    4   43]
 [   9   12   11   32    8   44   14   17  810   17]
 [  11    8    4   10   34   16    1   24    4  897]]

```
![mnist_linearreg_confusion_matrix](https://cloud.githubusercontent.com/assets/22183834/24684626/2f421872-195c-11e7-8c83-099154f83e2b.png)

The acuracy of Linear Regression is `90.1%`. Here is some of the wrong predictions of the Linear Regression:
![mnist_linearreg_wrongpred](https://cloud.githubusercontent.com/assets/22183834/24684617/297d96fa-195c-11e7-8395-7e2cf936f32b.png)

#### MLP Model:
The NN model is:
```
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
```
weights in MLP are not interpretable and I didn't show them here. The accuracy of MLP can be vary by altering the number of layers and the number of neurons at each layer (known as hyper parameter). 

#### RNN Model:
The RNN model with LSTM cell:
```
def recurrent_neural_network_model(x):
	
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
	                  'biases':tf.Variable(tf.random_normal([n_classes]))}
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	return output
```

In the `28*28` image, each row of image considered as one chunck size feeded to LSTM cell. 
Accuracy obove 98% is achived. The accuracy was lower than CNN, but the simulation was faster. 

#### CNN Model:

2D convolution layer is constructed as:
```
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
```

The weights of 16 `5*5` filters for first layer: 
![mnist_cnn_layer1_filter_weight](https://cloud.githubusercontent.com/assets/22183834/24685431/4d8a47ae-1962-11e7-9a6b-acd2354e1401.png)

One output example of first layer (after `2*2` pooling):
![mnist_cnn_layer1out_image](https://cloud.githubusercontent.com/assets/22183834/24686068/4b2721b8-1966-11e7-8258-b717c698e83f.png)

One output example of second layer (after `2*2` pooling):
![mnist_cnn_layer2out_image](https://cloud.githubusercontent.com/assets/22183834/24686069/4c6ff13a-1966-11e7-9366-93a23c23a2a2.png)

CNN reaches to an accuracy of 99%. 
CNN had the highest accuracy among the four other methods in the xpemces of longer runtime and limitation in memory usage for batch sizes greather 256 (for my laptop with 8 GB RAM).

For a better demonstration of CNN please see the [Magnus Pedersen's handout](https://github.com/Hvass-Labs/TensorFlow-Tutorials).










