import pickle
import gzip
import numpy as np 
import collections
from nltk.stem import WordNetLemmatizer
# ============ End Imports ============
class DataSet():
	def __init__(self,corpus,Train,Test,labels_type,Reverse_pairs):
		self.corpus=corpus
		self.Train_dataset=Train[0]
		self.Train_pairs_file=Train[1]
		self.Test_dataset=Test[0]
		self.Test_pairs_file=Test[1]
		self.lemmatizer = WordNetLemmatizer()
		self.labels_type=labels_type
		self.Reverse_pairs=Reverse_pairs
		if self.labels_type=='gold':
			self.read_pairs()
		elif self.labels_type=='proxy':
			self.read_proxy_labelled_pairs()
		pass
	# ------------------------------------------
	def Retrieve_Patterns(self,id2Patterns,Patterns_per_pair):
		print ("Retrieving Patterns")
		self.id2Patterns=load_zipped_pickle(id2Patterns)
		self.Remove_Placeholders() # To remove X and Y placeholders from the patterns
		self.Patterns2id={v:k for k,v in self.id2Patterns.items()}
		self.Patterns_per_pair_withFreq=load_zipped_pickle(Patterns_per_pair)
		self.compute_PatternsWeightsPerPair()
		self.Patterns_per_pair={pair:set(self.Patterns_per_pair_withFreq[pair]) for pair in self.Patterns_per_pair_withFreq}
		self.Training_triples=[(a,b,self.id2Patterns[p]) for (a,b) in self.Patterns_per_pair for p in self.Patterns_per_pair[(a,b)] if len(self.Patterns_per_pair[(a,b)])!=0]
		self.Pattern_Maximum_Length()
	# -------------------------------------------------------
	def compute_PatternsWeightsPerPair(self):
		self.PatternsWeightsPerPair={}
		for (a,b) in self.Patterns_per_pair_withFreq:
			Freq=collections.Counter(self.Patterns_per_pair_withFreq[a,b])
			Total=np.sum([Freq[key] for key in Freq])
			for p_id in Freq:
				pattern=self.id2Patterns[p_id]
				w=float(Freq[p_id])/Total
				self.PatternsWeightsPerPair[(a,b,pattern)]=w
	# -------------------------------------------------------
	def Remove_Placeholders(self):
		for p_id in self.id2Patterns:
			pattern=self.id2Patterns[p_id]
			pattern=pattern.strip().split(' ')
			pattern.remove('X')
			pattern.remove('Y')
			self.id2Patterns[p_id]=' '.join(pattern)
	# -------------------------------------------------------	
	def read_pairs(self):
		self.Train_Pairs=set()
		self.Pair2Rel={}

		with open(self.Train_pairs_file) as pairs:
			for line in pairs:
				if line.startswith(":"):
					rel=line.strip().split(':')[1]
				else:
					p = line.strip().split()
					(a, b) = p
					a_,b_=a.lower(),b.lower()
					self.Train_Pairs.add((a_,b_))
					self.Pair2Rel[(a_,b_)]=rel+'<'
					if self.Reverse_pairs:
						self.Pair2Rel[(b_,a_)]=rel+'>'
						
		print ("Number of pairs in %s:"%self.Train_dataset, len(self.Train_Pairs))

		self.Test_Pairs=set()
		with open("../"+self.Test_pairs_file) as pairs:
			for line in pairs:
				if line.startswith(":"):
					rel=line.strip().split(':')[1]
				else:
					p = line.strip().split()
					(a, b) = p
					a_,b_=a.lower(),b.lower()
					self.Test_Pairs.add((a_,b_))
						
		print ("Number of pairs in %s:"%self.Test_dataset, len(self.Test_Pairs))
	# -------------------------------------------------------	
	def read_proxy_labelled_pairs(self):
		self.Train_Pairs=set()
		self.Pair2Rel={}
		with open("../DiffVec_Pseudo_Labelled.txt") as pairs:
			for line in pairs:
				if line.startswith(":"):
					rel=line.strip().split(':')[1]
				else:
					p = line.strip().split()
					(a, b) = p
					a_,b_=a.lower(),b.lower()
					self.Train_Pairs.add((a_,b_))
					self.Pair2Rel[(a_,b_)]=rel+'<'
					if self.Reverse_pairs:
						self.Pair2Rel[(b_,a_)]=rel+'>'	
		print ("Number of pairs in %s:"%self.Train_dataset, len(self.Train_Pairs))

		self.Test_Pairs=set()
		with open("../"+self.Test_pairs_file) as pairs:
			for line in pairs:
				if line.startswith(":"):
					rel=line.strip().split(':')[1]
				else:
					p = line.strip().split()
					(a, b) = p
					a_,b_=a.lower(),b.lower()
					self.Test_Pairs.add((a_,b_))
						
		print ("Number of pairs in %s:"%self.Test_dataset, len(self.Test_Pairs))

	# -------------------------------------------------------
	def Filtering_Patterns(self,WR):
		# Remove triples with OOV_list words
		OOV_list=[]
		for (a,b,p) in self.Training_triples:
			words=set()
			words.add(a)
			words.add(b)
			for w in p.strip().split(' '):
				words.add(w)
			if not np.all([w in WR.vects for w in words]):
				OOV_list.append((a,b,p))
		for triple in OOV_list:
			self.Training_triples.remove(triple)
		print ("Number of training triples after removing OOV words:",len(self.Training_triples))
	# ------------------------------------------
	def Generate_Embedding_Matrix(self,WR):
		self.word2id={}
		self.word2id['<PAD>']=0
		self.word2id['X']=1
		self.word2id['Y']=2
		word_id=3

		# Assign ids to the words in word-pairs set
		Word_pairs=self.Train_Pairs.union(self.Test_Pairs)
		for (a,b) in Word_pairs:
			if a not in self.word2id:
				self.word2id[a]=word_id
				word_id+=1
			if b not in self.word2id:
				self.word2id[b]=word_id
				word_id+=1
		# Assign ids to the words in the lexical-pattersn
		self.Filtering_Patterns(WR) #Remove training instances with OOV words
		for (a,b,p) in self.Training_triples:
			for word in p.strip().split(' '):
				if word not in self.word2id:
					self.word2id[word]=word_id
					word_id+=1

		# Ea=np.random.rand(len(self.word2id),WR.dim).astype('float32')
		Ea=np.random.normal(size=(len(self.word2id),WR.dim)).astype('float32')
		# Ea=np.random.uniform(size=(len(self.word2id),WR.dim)).astype('float32')
		print ("Ea embedding matrix shape:",Ea.shape)
		c=0
		for word,word_id in self.word2id.items():
			if word in WR.vects:
				Ea[word_id]=WR.vects[word]
			elif word.lower() in WR.vects:
				Ea[word_id]=WR.vects[word.lower()]
			else:
				c+=1
		print ("Number of words without embeddings=",c)
		
		self.id2word={v:k for k,v in self.word2id.items()}

		# map training triples of (a,b,p) form to (a_id,b_id,p_id)
		self.Training_triplesIDs=[]
		if self.labels_type=='gold':
			for i,(a,b,p) in enumerate(self.Training_triples):
				# Filters: remove any pair that appear in the testing set and only consider semantic relations to train
				# and self.Pair2Rel[(a,b)].startswith('sem')
				if (a,b) not in self.Test_Pairs and (a,b) in self.Pair2Rel:
					a_id,b_id,p_id=self.word2id[a],self.word2id[b],self.Patterns2id[p]
					w=self.PatternsWeightsPerPair[(a,b,p)]
					rel=self.Pair2Rel[(a,b)]
					self.Training_triplesIDs.append((a_id,b_id,p_id,w,rel))
		elif self.labels_type=='proxy':
			for i,(a,b,p) in enumerate(self.Training_triples):
				# Filters: remove any pair that appear in the testing set and only consider semantic relations to train
				if (a,b) not in self.Test_Pairs and (a,b) in self.Pair2Rel:
					a_id,b_id,p_id=self.word2id[a],self.word2id[b],self.Patterns2id[p]
					w=self.PatternsWeightsPerPair[(a,b,p)]
					rel=self.Pair2Rel[(a,b)]
					self.Training_triplesIDs.append((a_id,b_id,p_id,w,rel))

		self.Used_PatternsIDs=list(set([t[2] for t in self.Training_triplesIDs]))
		print ("Number of training triples:",len(self.Training_triplesIDs))
		print ("Number of used patterns:",len(self.Used_PatternsIDs))
		return Ea
	# -------------------------------------------------------
	def Pattern_Maximum_Length(self):
		self.max_length=0
		patterns_in_train=set([t[2] for t in self.Training_triples])
		for pattern in patterns_in_train:
			length=len(pattern.strip().split(' '))
			if length>self.max_length:
				self.max_length=length
		print ("The maximum sequence length for the patterns is:",self.max_length)
#------------------------------------------------------------------
#  ============ End of the DataSet class ============
def load_zipped_pickle(filename):
	with gzip.open(filename,'rb') as f:
		loaded_object=pickle.load(f)
		return loaded_object
# -------------------------------------------------------

