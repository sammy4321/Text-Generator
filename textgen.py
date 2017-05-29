from __future__ import print_function	
import numpy as np
import tensorflow as tf
import dill

#training_text_features="abcdefghijklmnopqrstuvwxyz"
training_text_features=open('wiki.test.tokens').read()
all_chars=list(set(training_text_features))
print(all_chars)

training_text_labels=training_text_features+'.'
training_text_labels=training_text_labels[1:]
#print(training_text_features)
#print(training_text_labels)

len_of_features=len(training_text_features)
print(len_of_features)
n_inputs = len(all_chars) 
batch_len=100
total_no_of_batches=len_of_features/batch_len
print(total_no_of_batches)

def ret_one_hot(x):
	val=np.zeros([n_inputs])
	pos=-1
	for i in range(n_inputs):
		if x is all_chars[i]:
			pos=i
			break
	val[i]=1.0
	return(val)

training_one_hot_features=[]
training_one_hot_labels=[]

#for i in range(len_of_features):
#	training_one_hot_features.append(ret_one_hot(training_text_features[i]))
#training_one_hot_features=np.array(training_one_hot_features)

#training_one_hot_features=np.flip(training_one_hot_features,1)

#for i in range(len_of_features):
#	training_one_hot_labels.append(ret_one_hot(training_text_labels[i]))
#training_one_hot_labels=np.array(training_one_hot_labels)

#training_one_hot_labels=np.flip(training_one_hot_labels,1)

#print(training_one_hot_features.shape)
#print(training_one_hot_labels.shape)

lr = 0.001
training_iters = 100000
batch_size = 1

  
n_steps = 1    
n_hidden_units = 1024
n_classes = n_inputs

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}    

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
next_char = tf.argmax(pred,1)
test_features=[]
for i in range(1):
	test_features.append(ret_one_hot('a'))
test_features=np.array(test_features)

#pred_char='i'

pre_string='this is the first time'

pred_char=pre_string[len(pre_string)-1]
pre_string=pre_string[:len(pre_string)-1]

print('prestring',pre_string)
print('pred_char',pred_char)
with tf.Session() as sess:
    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    hm_epochs=10
    step = 0
    for epoch in range(hm_epochs):
    	epoch_loss=0
    	global epoch_x
    	global epoch_y
    	for i in range(total_no_of_batches+1):
    		training_one_hot_features=[];training_one_hot_labels=[]
    		#print(i)

    		if i == total_no_of_batches:
    			for j in range(len_of_features%batch_len):
    				training_one_hot_features.append(ret_one_hot(training_text_features[batch_len*i+j]))
    				#print(training_text_features[batch_len*i+j],end='')
    			training_one_hot_features=np.array(training_one_hot_features)
    			#print('inside if')

    			for j in range(len_of_features%batch_len):
    				training_one_hot_labels.append(ret_one_hot(training_text_labels[batch_len*i+j]))
    			training_one_hot_labels=np.array(training_one_hot_labels)
    			epoch_x=training_one_hot_features.reshape([-1,n_steps,n_inputs])
    			epoch_y=training_one_hot_labels
    			#print(training_text_features[batch_len*i:-1])

    		else:
    			for j in range(batch_len):
    				training_one_hot_features.append(ret_one_hot(training_text_features[batch_len*i+j]))
    				#print(training_text_features[batch_len*i+j],end='')
    			training_one_hot_features=np.array(training_one_hot_features)
    			#print('Inside else')
    			for j in range(batch_len):
    				training_one_hot_labels.append(ret_one_hot(training_text_labels[batch_len*i+j]))
    			epoch_x=training_one_hot_features.reshape([-1,n_steps,n_inputs])
    			epoch_y=training_one_hot_labels
    			


    		_,c=sess.run([train_op,cost],feed_dict={x:epoch_x,y:epoch_y})
    		epoch_loss += c
    		

    		print('epoch ',epoch,' batch : ',i,' total : ',total_no_of_batches)
    			
    		
    	
    	
    	

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
    #print();print();print();print(sess.run(weights['in'][0][0]));print();print();print() 
    test_pre_String=[]
    for i in range(len(pre_string)):

   		test_pre_String.append(ret_one_hot(pre_string[i]))

    test_pre_String=np.array(test_pre_String)
    print(pre_string,end='')
    print(pred_char,end='')

    sess.run([pred],feed_dict={x:test_pre_String.reshape([-1,n_steps,n_inputs])})
    for i in range(100):
    	test_features=[]
    	for i in range(1):
    		test_features.append(ret_one_hot(pred_char))
    	test_features=np.array(test_features)
    	
    	pred_val=sess.run([next_char],feed_dict={x:test_features.reshape([-1,n_steps,n_inputs])})
    	num_val=pred_val[0][0]
    	#print(a[0][0])
    	print(all_chars[num_val],end='')
    	pred_char=all_chars[num_val]
    	#if pred_char is '.' :
    	#	break
    W_w=sess.run(weights)
    B_b=sess.run(biases)

print()
class rnnNetwork:
	pass
rnet=rnnNetwork()
rnet.wts=W_w
rnet.bs=B_b
with open('weights_biases_saved.pkl','wb') as f:
	dill.dump(rnet,f)
print('saving Finished')

