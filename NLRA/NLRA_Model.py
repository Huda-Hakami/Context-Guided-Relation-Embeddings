import tensorflow as tf 
import re
import sklearn.preprocessing
# ============ End Imports ============
class NLRA():
	def __init__(self,activ,batch_size):
		self.Learning_Rate=0.01
		self.activ=activ
		self.batch_size=batch_size
#------------------------------------------------------------
	def G1_model(self,Ea,G1_BN,G1_HL,G1_Hdim,G1_l2_reg):
		'''
		Generator Net 1 (G1): MLP for word-embeddings
		'''
		# Hyperparameters
		self.G1_BN=G1_BN # boolean variable indicate batch normalization on G1 MLP generator
		self.Num_G1_hidden_layers=G1_HL
		self.G1_Hdim=G1_Hdim
		self.G1_l2_reg=G1_l2_reg
		if self.G1_BN:
			self.Moving_Means={}
			self.Moving_Variance={}
		self.Ea =tf.Variable(Ea,trainable=True,name='Ea')
		# constraint=lambda t: tf.clip_by_norm(t, 1.0, axes == [1])
		self.a_ids=tf.placeholder(tf.int32,[None])
		self.b_ids=tf.placeholder(tf.int32,[None])
		a_e=tf.nn.embedding_lookup(self.Ea,self.a_ids)
		b_e=tf.nn.embedding_lookup(self.Ea,self.b_ids)
		self.G1_pkeep=tf.placeholder(tf.float32,name="G1_pkeep") # probability to keep neurones while applying dropout regularization on G1
		self.is_training = tf.placeholder(tf.bool)

		# Automatically generates weights,bias and output of each G1_hidden layer in G1

		# G1_input_layer=tf.concat([a_e,b_e],1)
		G1_input_layer=tf.concat([a_e,b_e,tf.subtract(b_e,a_e)],1)
		G1_InputDim=int(G1_input_layer.get_shape()[1])

		G1_neurons=[G1_InputDim]
		for i in range(self.Num_G1_hidden_layers):
			G1_neurons.append(self.G1_Hdim)

		self.G1_hidden={}
		for layer in range(1,self.Num_G1_hidden_layers+1):
			weight_shape=[G1_neurons[layer-1],G1_neurons[layer]]
			bias_shape=[G1_neurons[layer]]
			self.G1_hidden['W%d'%layer]=variable(weight_shape,'W%d'%layer)
			self.G1_hidden['b%d'%layer]=variable(bias_shape,'b%d'%layer)
		self.Last_G1_output=self.Feed_pair_toMLP('G1',self.Num_G1_hidden_layers,self.G1_hidden,G1_input_layer,self.G1_BN,self.G1_pkeep)

		# self.Last_G1_output=tf.subtract(b_e,a_e)
		print ('last_G1_output shape:',self.Last_G1_output.get_shape())
		self.G1_l2_regularizer=tf.nn.l2_loss(self.G1_hidden['W1'])+tf.nn.l2_loss(self.G1_hidden['b1'])
		for i in range(2,self.Num_G1_hidden_layers+1):
			self.G1_l2_regularizer+=tf.nn.l2_loss(self.G1_hidden['W%d'%i])+tf.nn.l2_loss(self.G1_hidden['b%d'%i])
#------------------------------------------------------------
	def G2_rnn_model(self,Seq_Max_Leng,G2_HL,G2_Hdim):	
		'''
		Generator Net 2 (G2): LSTM for patterns
		'''
		# Hyperparameters
		self.Num_G2_hidden_layers=G2_HL
		self.G2_Hdim=G2_Hdim
		self.G2_pkeep=tf.placeholder(tf.float32,name="G2_pkeep") 

		self.Seq_Max_Leng=Seq_Max_Leng+2

		self.patterns_ids=tf.placeholder(tf.int32,[None,self.Seq_Max_Leng])
		self.early_stop=tf.placeholder(tf.int32,[None])
		sequence=tf.nn.embedding_lookup(self.Ea,self.patterns_ids)
		print (sequence.get_shape())

		rnn_cell=tf.contrib.rnn.BasicLSTMCell(self.G2_Hdim)
		rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.G2_pkeep)
        # rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * self.Num_G2_hidden_layers)
		self.G2_output, G2_state = tf.nn.dynamic_rnn(rnn_cell,sequence,dtype=tf.float32,sequence_length=self.early_stop)
		# print (self.G2_output.get_shape())
		# print (G2_state[0].get_shape())
		# print (G2_state[1].get_shape())
		# exit()
		self.G2_last = last_relevant(self.G2_output,self.early_stop)
#------------------------------------------------------------
	def define_NLRA_loss(self):	
		self.Y_=tf.placeholder(tf.float32,[None]) #True labels
		Dot_Product=tf.reduce_sum(tf.multiply(self.Last_G1_output,self.G2_last),1)
		# self.loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_, logits=Dot_Product))
		pos_loss =-tf.log(tf.nn.sigmoid(Dot_Product))
		neg_loss =-tf.log(tf.nn.sigmoid(-Dot_Product))
		self.loss = tf.divide(tf.reduce_sum(tf.multiply(self.Y_, pos_loss) + tf.multiply(1. - self.Y_, neg_loss)),self.batch_size)
		# self.loss=tf.reduce_mean(self.loss+self.G1_l2_reg*self.G1_l2_regularizer)	
#------------------------------------------------------------
	def optimize(self):
		batch = tf.Variable(0, trainable=False)
		optimizer=tf.train.AdagradOptimizer(self.Learning_Rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_step=optimizer.minimize(self.loss,global_step=batch)
#------------------------------------------------------------
	def Feed_pair_toMLP(self,Net,Num_Hidden_layers,hidden_weights,input_layer,BN,pkeep):
		output={}
		for layer in range(1,Num_Hidden_layers+1):
			if layer==1:
				inp=input_layer
			else:
				inp=output['Y%d'%(layer-1)]
			# compute activations of the G1_hidden layer
			if BN:
				output['Y%d'%layer]=self.hidden_layer_output_bn(inp,hidden_weights['W%d'%layer],'%slayer%d'%(Net,layer),layer)
			else:
				output['Y%d'%layer]=self.hidden_layer_output(inp,hidden_weights['W%d'%layer],hidden_weights['b%d'%layer],pkeep)
		return output['Y%d'%Num_Hidden_layers]
#------------------------------------------------------------------	
	def hidden_layer_output(self,X,W,b,pkeep):
		if self.activ=='sigmoid':
			return tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(X,W),b)),pkeep)
		elif self.activ=='tanh':
			return tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(X,W),b)),pkeep)
		elif self.activ=='relu':
			return tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X,W),b)),pkeep)
		elif self.activ=='linear':
			return tf.nn.dropout(tf.add(tf.matmul(X,W),b),self.pkeep)
		else:
			raise ValueError
#------------------------------------------------------------------	
	def hidden_layer_output_bn(self,X,W,scope,layer_id):
		with tf.variable_scope(scope):
			z=tf.matmul(X,W)
			bn_instance = tf.layers.BatchNormalization(trainable=True)
			batch_layer = bn_instance.apply(z,training=self.is_training)
			# Keep track for moving_means and moving_variances for the inferernce time
			self.Moving_Means[scope]=bn_instance.moving_mean
			self.Moving_Variance[scope]=bn_instance.moving_variance
			# print (self.Moving_Means)
			# print (self.Moving_Variance)
			if self.activ=='tanh':
				return tf.nn.tanh(batch_layer)
			elif self.activ=='relu':
				return tf.nn.relu(batch_layer)
			elif self.activ=='sigmoid':
				return tf.nn.sigmoid(batch_layer)
			else:
				raise ValueError
#------------------------------------------------------------------
	def Normalize_Ea(self):
		print ("Normalizing Ea matrix...")
		self.Ea=tf.nn.l2_normalize(self.Ea,axis=1)
#------------------------------------------------------------------
	def Normalize_Ec(self):
		print ("Normalizing Ec matrix...")
		self.Ec=tf.nn.l2_normalize(self.Ec,axis=1)
#  ============ End of the NLRA class ============
def variable(shape,var_name):
		return tf.Variable(tf.random_uniform(shape,minval=-1,maxval=1,dtype=tf.float32),name=var_name)
		# return tf.Variable(tf.truncated_normal(shape,dtype=tf.float32),name=var_name)
# -----------------------------------------------------------
def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length
# -----------------------------------------------------------
def last_relevant(output, length):
	batch_size = tf.shape(output)[0]
	max_length = tf.shape(output)[1]
	out_size = int(output.get_shape()[2])
	index = tf.range(0, batch_size) * max_length + (length - 1)
	flat = tf.reshape(output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return relevant
# -----------------------------------------------------------


