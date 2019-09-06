import numpy as np 
from wordreps import WordReps
from algebra import cosine, normalize
import tensorflow as tf 
import random
from dataset import DataSet
import NLRA_Model
from Eval import eval_SemEval
import sklearn.preprocessing
# ============ End Imports ============
class Training():
	def __init__(self):
		# Hyperparameters for compositional-based word pari embeddings (G1) 
		self.batchSize=100
		G1_HL=3
		G1_Hdim=WR.dim
		G1_BN=True #boolean variable T/F for batch normalization on G1 MLP 
		G1_l2_reg=0.0 # L2 regularization coefficient
		self.G1_pkeep=1.0 # 1.0 means no Dropout applied during training on G1

		# Hyperparameters for LSTM encoder for patterns (G2) 
		G2_HL=1
		G2_Hdim=WR.dim
		self.G2_pkeep=1.0 # 1.0 means no Dropout applied during training on G2
		activ='tanh'

		# Create relational model instance
		self.RelModel=NLRA_Model.NLRA(activ,self.batchSize)
		self.RelModel.G1_model(Ea,G1_BN,G1_HL,G1_Hdim,G1_l2_reg)
		self.RelModel.G2_rnn_model(DS.max_length,G2_HL,G2_Hdim)
		self.RelModel.define_NLRA_loss()
		self.RelModel.optimize()
		self.sess=tf.Session()
# --------------------------------------------------
	def Train_NLRA_Model(self):
		# Hyperparameters
		
		self.sess.run(tf.global_variables_initializer())
		n=10  #number of negative patterns for each word-pairs
		epochs=500
		winn_loss=1e7
		win_acc=-1

		Train = DS.Training_triplesIDs

		print ("Number of training triples (a,b,p)",len(Train))
		print ("==========================================================================")
		for epoch in range(epochs):
			# Randomly shuffle training instances for each epoch
			random.shuffle(Train)
			# performance  every 20 steps
			if epoch%1==0:
				Pair_Embeddings=self.Gen_Pair_Embeddings()
				acc_1,corr_1=eval_SemEval(Pair_Embeddings,'Test')
				acc_2,corr_2=eval_SemEval(Pair_Embeddings,'Valid')
				acc_3,corr_3=eval_SemEval(Pair_Embeddings,'All')
				print ("Epoch:%d, Acc_Test:%f, Acc_Valid:%f, Acc_All:%f, Corr_Test:%f, Corr_Valid:%f, Corr_All:%f"%(epoch,acc_1,acc_2,acc_3,corr_1,corr_2,corr_3))
				# For early stopping
				if acc_2>win_acc:
					win_acc=acc_2
					self.Save_Trained_Model()
					print ("Parameters and Pair-Embeddings are changed and saved...")
					best_epoch=epoch
					patient_cnt=0
				else:
					patient_cnt+=1
				if patient_cnt>10:
					print ("early stopping ... epoch number %d"%epoch)
					print ("Winner acc:%f at epoch:%d"%(win_acc,best_epoch))
			# Training
			for minibatch in next_batch(self.batchSize,Train):
				T_batch=Get_Pos_Neg_examples(minibatch,n)
				a_ids,b_ids,p_ids,labels=shred_tuples(T_batch)
				train_data={self.RelModel.a_ids:a_ids,self.RelModel.b_ids:b_ids,self.RelModel.G1_pkeep:self.G1_pkeep,self.RelModel.is_training:True}

				pattern_seq,early_stop=Pattern_Sequences_withTargetedEntities(a_ids,b_ids,p_ids)
				train_data[self.RelModel.patterns_ids]=pattern_seq
				train_data[self.RelModel.early_stop]=early_stop
				train_data[self.RelModel.G2_pkeep]=self.G2_pkeep
				train_data[self.RelModel.Y_]=labels

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

		a_ids=[t[1] for t in word_pairs_ids]
		b_ids=[t[0] for t in word_pairs_ids]
		dic={self.RelModel.a_ids:a_ids,self.RelModel.b_ids:b_ids,self.RelModel.G1_pkeep:1.0,self.RelModel.is_training:False}
		Pair_Embeddings2=self.sess.run(self.RelModel.Last_G1_output,feed_dict=dic)

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
def Get_Pos_Neg_examples(batch,n):
	# print ("Generating negative examples ...")
	T_batch=[]
	PATTERNS=[p for (a,b,p) in batch]
	for i,(a,b,p) in enumerate(batch):
		# Generate negative triple
		T_batch.append((a,b,p,1.0))
		negative_triples=[]
		for i in range(n):
			random_pattern=random.sample(PATTERNS,1)[0]
			while random_pattern in DS.Patterns_per_pair[(DS.id2word[a],DS.id2word[b])]:
				random_pattern=random.sample(PATTERNS,1)[0]
			negative_triples.append((a,b,random_pattern)) 
		for (a_,b_,p_) in negative_triples:
			T_batch.append((a_,b_,p_,0.0))
	return T_batch
# -------------------------------------------------------
def shred_tuples(tuples):
	a_ids=[t[0] for t in tuples]
	b_ids=[t[1] for t in tuples]
	p_ids=[t[2] for t in tuples]
	label=[t[3] for t in tuples]
	return a_ids,b_ids,p_ids,label
# -------------------------------------------------------	
def Pattern_Sequences(p_ids):
	# pattern_seq=[[0 for j in range(DS.max_length)] for i in range(len(p_ids))]
	pattern_seq=np.zeros((len(p_ids),DS.max_length),dtype=int) #+2 is for the targeted two entities a and b
	early_stop=[]
	for i in range(len(p_ids)):
		pattern=DS.id2Patterns[p_ids[i]]
		words=pattern.strip().split(' ')
		early_stop.append(len(words))
		for j,w in enumerate(words):
			pattern_seq[i,j]=DS.word2id[w]
	return pattern_seq,early_stop
# -------------------------------------------------------	
def Pattern_Sequences_withTargetedEntities(a_ids,b_ids,p_ids):
	# pattern_seq=[[0 for j in range(DS.max_length)] for i in range(len(p_ids))]
	pattern_seq=np.zeros((len(p_ids),DS.max_length+2),dtype=int) #+2 is for the targeted two entities a and b
	early_stop=[]
	for i in range(len(p_ids)):
		pattern=DS.id2Patterns[p_ids[i]]
		words=pattern.strip().split(' ')
		words.insert(0,DS.id2word[a_ids[i]])
		words.append(DS.id2word[b_ids[i]])
		early_stop.append(len(words))
		for j,w in enumerate(words):
			pattern_seq[i,j]=DS.word2id[w]
	return pattern_seq,early_stop
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
	WR.vects['X']=np.random.normal(size=(WR.dim)).astype('float32')
	WR.vects['Y']=np.random.normal(size=(WR.dim)).astype('float32')

	'''
	Dataset 
	'''
	corpus='Wikipedia_English'
	Train_dataset=('DiffVec',"../DiffVec_Pairs")
	Test_dataset=('SemEval',"../SemEval_Pairs.txt")

	id2Patterns="../Relational_Patterns/Patterns_Xmid5Y"
	Patterns_per_pair="../Relational_Patterns/Patterns_Xmid5Y_PerPair"

	DS=DataSet(corpus,Train_dataset,Test_dataset)
	DS.Retrieve_Patterns(id2Patterns,Patterns_per_pair)
	DS.read_pairs()
	DS.Pattern_Maximum_Length()
	Ea=DS.Generate_Embedding_Matrix(WR)
	
	'''
	Training & Evaluation 
	'''
	Eval=Training()
	Eval.Train_NLRA_Model()


