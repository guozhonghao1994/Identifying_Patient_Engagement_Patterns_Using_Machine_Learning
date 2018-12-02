# main
import tensorflow as tf
import numpy as np

def read_data(filename1,filename2):
	
	training_data = np.load(filename1)
	testing_data = np.load(filename2)
	testing_data[:,-1] = 0
	train_data = scale(training_data[:,:-1].astype(float))
	test_data = scale(testing_data[:,:-1].astype(float))
	train_label = training_data[:,-1].reshape(-1,1)
	test_label = testing_data[:,-1].reshape(-1,1)
	return train_data,test_data,train_label,test_label

	
def scale(X):
	X-=np.mean(X,axis=0)
	X/=np.std(X,axis=0)
	return X


def network(x):
	fc_1 = tf.layers.dense(inputs=x,units=16,activation=tf.nn.relu,name='fc_1')
	fc_2 = tf.layers.dense(inputs=fc_1,units=8,activation=tf.nn.relu,name='fc_2')
	logits = tf.layers.dense(inputs=fc_2,units=2,activation=None,name='logits')
	return logits


# load data
train_data,test_data,train_label,test_label = read_data('train_with_message.npy','test_with_message.npy')
feature_num = int(train_data.shape[1])
print(train_data.shape,test_data.shape)

x = tf.placeholder(tf.float32,[None,feature_num])
y_ = tf.placeholder(tf.int32,[None,1])

logits = network(x)
y_dict = dict(labels=y_,logits=logits)
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(**y_dict))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_loss)

y_pred = tf.argmax(tf.nn.softmax(logits),axis=1)
y_true = tf.argmax(y_,axis=1)
correct_prediction = tf.equal(tf.cast(y_pred,tf.int64),y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(200):
		train_step.run(feed_dict={x:train_data,y_:train_label})
		train_accuracy = accuracy.eval(feed_dict={x:train_data,y_:train_label})
		test_accuracy = accuracy.eval(feed_dict={x:test_data,y_:test_label})
		print('step %d, train_accuracy %g, test_accuracy %f' % (i,train_accuracy,test_accuracy))
