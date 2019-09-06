import numpy as np
Pair_Embeddings={}
path='Pretrained word-pairs Embeddings/SemEval_CGRE_Proxy.npy'
loaded_object=np.load(path,allow_pickle=True).item()
for (a,b) in loaded_object:
	Pair_Embeddings[(a,b)]=loaded_object[(a,b)]
print ("Number of word-pairs:",len(Pair_Embeddings))