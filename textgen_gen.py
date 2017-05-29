from __future__ import print_function
import tensorflow as tf
import numpy as np
import dill
with open('weights_biases_saved.pkl','rb') as f:
	rnet=dill.load(f)

print("Loading Complete")
print(rnet.wts['in'][0][0])

def ret_one_hot(x):
	val=np.zeros([128])
	val[ord(x)]=1.0
	return(val)



lr = 0.001
training_iters = 100000
batch_size = 1

n_inputs = 128   
n_steps = 1    
n_hidden_units = 1024
n_classes = 128

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(rnet.wts['in']),
    'out': tf.Variable(rnet.wts['out'])}
biases = {
    'in': tf.Variable(rnet.bs['in']),
    'out': tf.Variable(rnet.bs['out'])}    

def RNN(X, weights, biases):
    
    X = tf.reshape(X, [-1, n_inputs])

    
    X_in = tf.matmul(X, weights['in']) + biases['in']
    
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

   

    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(n_hidden_units)
    
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in,time_major=False,dtype=tf.float32)

    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results



pred = RNN(x, weights, biases)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#train_op = tf.train.AdamOptimizer().minimize(cost)

#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
next_char = tf.argmax(pred,1)
test_features=[]
for i in range(1):
	test_features.append(ret_one_hot('a'))
test_features=np.array(test_features)

pred_char='x'
pre_string='abcdefghijklmnopqrstuvw'
with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)
	test_pre_String=[]
	for i in range(len(pre_string)):
			test_pre_String.append(ret_one_hot(pre_string[i]))
	test_pre_String=np.array(test_pre_String)
	sess.run([pred],feed_dict={x:test_pre_String.reshape([-1,n_steps,n_inputs])})

	for i in range(26):
		test_features=[]
		
		
		for i in range(1):
			test_features.append(ret_one_hot(pred_char))
		test_features=np.array(test_features)

		pred_val=sess.run([next_char],feed_dict={x:test_features.reshape([-1,n_steps,n_inputs])})
		num_val=pred_val[0][0]
		print(chr(num_val),end='')
		pred_char=chr(num_val)
		if pred_char is '.':
			break
print()