import numpy as np 
from wordreps import WordReps
from algebra import cosine, normalize
import tensorflow as tf 
import random
from dataset import DataSet
import CGRE_Model
from Eval import eval_SemEval
import sklearn.preprocessing
# ============ End Imports ============
class Training():
	def __init__(self):
		# Compositional relation embeddings (G1) Hyperparameters
		self.batchSize=100
		G1_HL=3
		G1_Hdim=WR.dim
		G1_BN=True #boolean variable T/F for batch normalization on G1 MLP 
		G1_l2_reg=0.001 # L2 regularization coefficient
		self.G1_pkeep=1.0 # 1.0 means no Dropout applied during training on G1

		# LSTM pattern encoding (G2) Hyperparameters
		G2_HL=1
		G2_Hdim=WR.dim
		self.G2_pkeep=1.0 # 1.0 means no Dropout applied during training on G2
		activ='tanh'

		# Create relational model instance
		self.RelModel=CGRE_Model.CGRE(activ,self.batchSize)
		self.RelModel.G1_model(Ea,G1_BN,G1_HL,G1_Hdim,G1_l2_reg)
		self.RelModel.G2_rnn_model(DS.max_length,G2_HL,G2_Hdim)
# --------------------------------------------------
	def Train_Model(self):
		# Hyperparameters
		epochs=500
		hist_loss=[]
		hist_acc=[]
		winn_loss=1e7
		win_acc=-1
		# Discriminator Hyperparameters (for Rel-Rep-alignment model)
		D_HL=0
		D_Hdim=WR.dim
		D_BN=False # boolean variable T/F for batch normalization on D
		self.D_pkeep=1.0 # 1.0 means no Dropout applied during training on the Discriminator D
		D_l2_reg=0.001 # L2 regularization coefficient (to perform l2 regularized cross-entropy)

		Train = DS.Training_triplesIDs
		
		Train_Relations=set([rel for (a,b,p,w,rel) in Train])
		Num_of_Classes=len(Train_Relations)
		print ("Number of relation labels for cross-entropy objective=",Num_of_Classes)
		# Assign ids to relations
		Rel2id={}
		i=0
		for rel in Train_Relations:
			Rel2id[rel]=i
			i+=1
		Train_dic={}
		for (a,b,p,w,rel) in Train: 
			Train_dic.setdefault((a,b,rel),[])
			Train_dic[(a,b,rel)].append((p,w))

		Training_patterns=set([p for (_,_,p,_,_) in Train])
		print ('Number of training patterns after removing test instances=',len(Training_patterns))

		Train_list=list(Train_dic.keys())
		print ("Number of training word-pairs (a,b,[(p,w)])",len(Train_list))

		self.RelModel.define_loss(D_HL,D_Hdim,D_BN,D_l2_reg,Num_of_Classes)
		self.RelModel.optimize()
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
		print ("==========================================================================")
		for epoch in range(epochs):
			# Randomly shuffle training instances for each epoch
			random.shuffle(Train_list)
			# performance  every 20 steps
			if epoch%1==0:
				Pair_Embeddings=self.Gen_Pair_Embeddings()
				acc_1,corr_1=eval_SemEval(Pair_Embeddings,'Test')
				acc_2,corr_2=eval_SemEval(Pair_Embeddings,'Valid')
				acc_3,corr_3=eval_SemEval(Pair_Embeddings,'All')
				print ("Epoch:%d, Acc_Test:%f, Acc_Valid:%f, Acc_All:%f, Corr_Test:%f, Corr_Valid:%f, Corr_All:%f"%(epoch,acc_1,acc_2,acc_3,corr_1,corr_2,corr_3))

				hist_acc.append(acc_2)
				# For early stopping
				if acc_2>win_acc:
					win_acc=acc_2
					self.Save_Trained_Model()
					print ("Parameters and Pair-Embeddings are changed...")
					best_epoch=epoch
					patient_cnt=0
				else:
					patient_cnt+=1
				if patient_cnt>10:
					print ("early stopping ... epoch number %d"%epoch)
					print ("Winner acc:%f at epoch:%d"%(win_acc,best_epoch))
					# break
			# Training
			for minibatch in next_batch(self.batchSize,Train_list):
				a_ids,b_ids,labels=shred_tuples(minibatch)
				Train_Y=np.zeros((len(minibatch),Num_of_Classes))
				for i,rel in enumerate(labels):
					rel_id=Rel2id[rel]
					Train_Y[i,rel_id]=1.0

				train_data={self.RelModel.a_ids:a_ids,self.RelModel.b_ids:b_ids,self.RelModel.G1_pkeep:self.G1_pkeep,\
							self.RelModel.is_training:True,self.RelModel.D_pkeep:self.D_pkeep}

				minibatch_patterns=[Train_dic[(a,b,rel)] for (a,b,rel) in minibatch]
				max_num_of_patterns,pattern_seq,early_stop,weights=Pattern_Sequences(a_ids,b_ids,minibatch_patterns)
				train_data[self.RelModel.max_num_of_patterns]=max_num_of_patterns
				train_data[self.RelModel.patterns_ids]=pattern_seq
				train_data[self.RelModel.early_stop]=early_stop
				train_data[self.RelModel.weights]=weights
				train_data[self.RelModel.G2_pkeep]=self.G2_pkeep

				# Loss options
				train_data[self.RelModel.Y_]=Train_Y
				self.sess.run(self.RelModel.train_step,feed_dict=train_data)		
# --------------------------------------------------
	def Save_Trained_Model(self):
		Pair_Embeddings_dic=self.Gen_Pair_Embeddings()
		np.save("res/Pair_Embeddings.npy",Pair_Embeddings_dic)
# --------------------------------------------------
	def Gen_Pair_Embeddings(self):
		
		word_pairs_ids=[(DS.word2id[a],DS.word2id[b]) for (a,b) in DS.Test_Pairs]
		a_ids=[t[0] for t in word_pairs_ids]
		b_ids=[t[1] for t in word_pairs_ids]
		dic={self.RelModel.a_ids:a_ids,self.RelModel.b_ids:b_ids,self.RelModel.G1_pkeep:1.0,self.RelModel.is_training:False}
		Pair_Embeddings1=self.sess.run(self.RelModel.Last_G1_output,feed_dict=dic)
		# Pair_Embeddings1=sklearn.preprocessing.normalize(Pair_Embeddings1,axis=1,norm='l2') #L2 norm of r(a,b)

		a_ids=[t[1] for t in word_pairs_ids]
		b_ids=[t[0] for t in word_pairs_ids]
		dic={self.RelModel.a_ids:a_ids,self.RelModel.b_ids:b_ids,self.RelModel.G1_pkeep:1.0,self.RelModel.is_training:False}
		Pair_Embeddings2=self.sess.run(self.RelModel.Last_G1_output,feed_dict=dic)
		# Pair_Embeddings2=sklearn.preprocessing.normalize(Pair_Embeddings2,axis=1,norm='l2') #L2 norm of r(b,a)

		Pair_Embeddings=np.hstack((Pair_Embeddings1,Pair_Embeddings2))

		Pair_Embeddings_dic={}
		for i,(a,b) in enumerate(DS.Test_Pairs):
			Pair_Embeddings_dic[(a,b)]=Pair_Embeddings[i]
		return Pair_Embeddings_dic
#  ============ End of the Evaluation class ============
def next_batch(batchSize,data):
	# loop over our dataset in mini-batches of size `batchSize`
	for i in np.arange(0, len(data), batchSize):
		# yield the current batched data 
		yield data[i:i + batchSize]
# -------------------------------------------------------
def shred_tuples(tuples):
	a_ids=[t[0] for t in tuples]
	b_ids=[t[1] for t in tuples]
	labels=[t[2] for t in tuples]
	return a_ids,b_ids,labels
# -------------------------------------------------------	
def Pattern_Sequences(a_ids,b_ids,minibatch_patterns):
	max_num_of_patterns=np.max([len(L) for L in minibatch_patterns])
	min_num_of_patterns=np.min([len(L) for L in minibatch_patterns])
	# print ("Max num of patterns:",max_num_of_patterns)
	# print ("Min num of patterns:",min_num_of_patterns)
	pattern_seq=np.zeros((len(a_ids)*max_num_of_patterns,DS.max_length+2),dtype=int) #+2 is for the targeted two entities a and b
	
	early_stop=[0 for i in range(len(a_ids)*max_num_of_patterns)]
	weights=[0.0 for i in range(len(a_ids)*max_num_of_patterns)]

	for i in range(len(a_ids)):
		set_of_patterns=minibatch_patterns[i]
		for j in range(max_num_of_patterns):
			if j<len(set_of_patterns):
				pattern_id,w=set_of_patterns[j][0],set_of_patterns[j][1]
				pattern=DS.id2Patterns[pattern_id]
				words=pattern.strip().split(' ')
				words.insert(0,DS.id2word[a_ids[i]])
				words.append(DS.id2word[b_ids[i]])
				early_stop[(i*max_num_of_patterns)+j]=len(words)
				weights[(i*max_num_of_patterns)+j]=w
				for k,word in enumerate(words):
					pattern_seq[(i*max_num_of_patterns)+j,k]=DS.word2id[word]

	return max_num_of_patterns,pattern_seq,early_stop,weights
# -----------------------------------------------------------	
if __name__=="__main__":
	'''
	Word Embeddings
	'''
	pretrained_glove_300=("../glove.6B.300d.zip","glove",300)
	WR=WordReps()
	norm=1
	standardise=0
	WR.Read_Embeddings_zip_file(pretrained_glove_300,norm,standardise)
	WR.vects['<PAD>']=np.zeros(WR.dim)
	# WR.vects['X']=np.random.rand(WR.dim)
	# WR.vects['Y']=np.random.rand(WR.dim)

	WR.vects['X']=np.random.normal(size=(WR.dim)).astype('float32')
	WR.vects['Y']=np.random.normal(size=(WR.dim)).astype('float32')

	'''
	Dataset 
	'''
	corpus='Wikipedia_English'
	Train_dataset=('DiffVec',"DiffVec_Pairs")
	Test_dataset=('SemEval',"SemEval_Pairs.txt")
	labels_type='proxy'
	Reverse_pairs=True


	DS=DataSet(corpus,Train_dataset,Test_dataset,labels_type,Reverse_pairs)
	id2Patterns="../Relational_Patterns/Patterns_Xmid5Y"
	Patterns_per_pair="../Relational_Patterns/Patterns_Xmid5Y_PerPair"
	DS.Retrieve_Patterns(id2Patterns,Patterns_per_pair)
	Ea=DS.Generate_Embedding_Matrix(WR)
	
	'''
	Training & Evaluation 
	'''
	Eval=Training()
	Eval.Train_Model()


