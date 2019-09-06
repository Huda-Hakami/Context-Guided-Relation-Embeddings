import numpy as np
import zipfile
from algebra import cosine, normalize

class WordReps():
	def __init__(self):
		self.vects=None
		self.vocab=None
		pass
#------------------------------------------------------------------
	def Read_Embeddings_zip_file(self,Embedding_file,norm,standardise):
		"""
		Read the word vectors where the first token is the word.
		"""
		self.embtype=Embedding_file[1]
		self.dim=Embedding_file[2]
		print ("Embedding Type:",self.embtype)
		print ("Embedding dim:",self.dim)

		vects = {}
		vocab = []
		print ("Retrieving words embeddings...")
		zfile = zipfile.ZipFile(Embedding_file[0])
		for finfo in zfile.infolist():
			F = zfile.open(finfo)
			# read the vectors.
			line = F.readline()
			if len(line.split())==2:
				print ("Header Exists.")
				line=F.readline()
			while len(line) != 0:
				p = line.split()
				word = p[0].decode('utf-8')
				v = np.zeros(self.dim, float)

				for i in range(0, self.dim):
					v[i] = float(p[i+1])
				# If you want to normalize the vectors, then call the normalize function.
				if norm:
					vects[word] = normalize(v)
				else:
					vects[word] = v
				vocab.append(word)
				line = F.readline()
			print ("Number of words in the vocabulary is: ",len(vocab))
			F.close()
			self.vocab = vocab
			self.vects = vects
			if standardise:
				self.Standardization()
			break
#------------------------------------------------------------------
	def Standardization(self):
		Embedding_matrix=np.zeros((len(self.vects),self.dim))
		i=0
		for word in self.vects:
			Embedding_matrix[i]=self.vects[word]
			i+=1
		mean=np.mean(Embedding_matrix,axis=0)
		sd=np.std(Embedding_matrix,axis=0)

		for word in self.vects:
			self.vects[word]=(self.vects[word]-mean)/(sd)
#------------------------------------------------------------