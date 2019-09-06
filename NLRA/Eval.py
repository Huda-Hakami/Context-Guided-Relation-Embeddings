import numpy as np 
from semeval import SemEval
from algebra import cosine, normalize
from dataset import DataSet
import operator
#------------------------------------------------------------------
def eval_SemEval(Pair_Embeddings,flag):
	"""
	Evaluate SemEval2012 Task 2 
	"""
	counter=0
	S = SemEval("../semeval")
	total_accuracy = 0.0
	total_correlations=0.0
	# print "Total no. of instances in SemEval =", len(S.data)
	Valid_Relations=["1a","2c","2h","3a","3c","4c","5d","5i","7a","10a"]
	if flag=='Valid':
		Relation_Set=Valid_Relations
	elif flag=='Test':
		Relation_Set=[Q['filename']for Q in S.data if Q['filename'] not in Valid_Relations]
	elif flag=='All':
		Relation_Set=[Q['filename']for Q in S.data]
	c=0
	Res_dic={}
	for Q in S.data:
		if Q['filename'] in Relation_Set:
			Res_dic.setdefault(Q['filename'],0.0)
			c+=1
			scores = []
			for (first, second) in Q["wpairs"]:
				val=0.0
				for (p_first, p_second) in Q["paradigms"]:
					#print first, second, p_first, p_second
					Rel_ab=Pair_Embeddings[(first.strip().lower(),second.strip().lower())]
					Rel_cd=Pair_Embeddings[(p_first.strip().lower(),p_second.strip().lower())]
					relsim=(cosine(Rel_ab,Rel_cd)+1.0)/2.0
					val += relsim
				val /= float(len(Q["paradigms"]))
				scores.append(((first, second), val))
				
			# sort the scores and write to a file. 
			# scores.sort(lambda x, y: -1 if x[1] > y[1] else 1)
			scores.sort(key=operator.itemgetter(1),reverse=True)
			score_fname = "work/semeval/%s.txt" % Q["filename"]
			score_file = open(score_fname, 'w')
			for ((first, second), score) in scores:
				score_file.write('%f "%s:%s"\n' % (score, first, second))
			score_file.close()
			ACC = S.get_accuracy(score_fname, Q["filename"])
			CORR= S.get_correlation(score_fname, Q["filename"])
			Res_dic[Q['filename']]=ACC
			total_accuracy+=ACC
			total_correlations+=CORR

	# print ("Number of tested relaitons:",c)
	acc = total_accuracy / float(c)
	corr= total_correlations / float(c)
	return acc,corr
#------------------------------------------------------------------



