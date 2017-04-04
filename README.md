# Applying TensorFlow for classifying [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)
In this attempt, I have applied TensorFlow (puthon 3.5 on Ubuntu) to construct the follwing four models:
1. Linear Regression with 1 fully connected layer
2. Multilayer Perceptron Model (MLP) with 3 fully connected layers
3. Recurrent Neural Network (RNN) with 1 NN layer inside a LSTM cell
4. Convolutional Neural Network (CNN) with 2 Convolutional/pooling layers and 1 fully connected layer


**Refrences:**
most of the materials are taken from:
* [TensorFlow Example for MNIST](https://www.tensorflow.org/get_started/mnist/pros)
* [Magnus Pedersen's tutorial](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
* [sentdex's tutorial channel](https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

please contact me of any questions/comments via: jamal.alikhani@gmail.com

## TensorFlow Architecture for Neural Network (general perspective)
1. **TensorFlow Graph Construction**
  1. Placeholder Variables:
  '
  x = tf.placeholder(tf.float32, [None, feature_size])
  y_true = tf.placeholder(tf.float32, [None, class_size]) 
  '
  
  2. Model Variables:
  '
  weights = tf.Variable(tf.zeros([feature_size, class_size]))
  biases = tf.Variable(tf.zeros([class_size]))
  '
  
  3. Model (logistic regression for example):
  '
  logits = tf.matmul(x, weights) + biases
  y_pred = tf.nn.softmax(logits)
  '
  
  4. Cost funtion:
  'cost_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true_onehot)
  cost = tf.reduce_mean(cost_entropy)'
  
  5. Optimizer algorithm:
  '''
  Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
  '''
  
  6. Model Performance:
  '''
  Correct_prediction = tf.equal(y_pred, y_true)
  accuracy = tf.reduce_mean(tf.cast(Correct_prediction, tf.float64))
  '''

2. **Running The TF Graph**
  1. Open a session:
  '''
  sess = tf.Session()
  ,,,
  
  2. initialization:
  '''
	sess.run(tf.initialize_all_variables())
  '''
  
  3. Training:
  '''
  batch_size = 256
	num_iteration = int(mnist.train.num_examples/batch_size)
	for i in range(num_iteration):
		x_batch, y_batch_onehot = mnist.train.next_batch(batch_size)
		feed_dict_train = {x:x_batch, y_true_onehot:y_batch_onehot}
		sess.run(Optimizer,feed_dict=feed_dict_train)
		w = sess.run(weights)
  '''
  
  4. Accuracy evaluation:
  '''
	feed_dict_test = {x:mnist.test.images, y_true_onehot:mnist.test.labels, y_true_int:mnist.test.cls}
	acc = sess.run(accuracy, feed_dict = feed_dict_test)
  '''
	5. weights show
  '''
	w = sess.run(weights)
  '''
  
  6. Close the session:
  '''
  sess.close()
  '''



