# CGRE: Context-Guided-Relation-Embeddings
This directory contains datasets and implementations of context-guided relation embedding (CGRE) proposed in our paper: Context-guided Self-Supervised Relation Embeddings.

It also implement neural latent relational analysis (NLRA) for word-pair representations, a method proposed by Washio and Kato, 2018 in their paper: Neural Latent Relational Analysis to Capture Lexical Semantic Relations in a Vector Space.

We used python 3.6.1.
These codes require tensorflow, sklearn and numpy.
This software includes the work that is distributed in the Apache License 2.0.
# Data
This project contains the following data files:
- Pretrained word-pairs Embeddings: the folder includes pre-trained word-pair embeddings for SemEval-2012 Task2. Embeddings for four different methods are available as follows:

        1. SemEval_CGRE_Gold.npy: supervised method trained on gold relation labels for DiffVec data in DiffVec_Pairs file
        
        2. SemEval_CGRE_Proxy.npy: self-supervised method trained on pseudo labels of DiffVec in DiffVec_Pseudo_Label.txt
        
        3. SemEval_MnnPL.npy: no contextual patterns are used to train Multi-class neural netword penultimate layer model. 
        
        4. SemEval_NLRA.npy: Neural Latent Relational Analysis word-pair embeddings 
        
- Read_PreTrained_WordPairs.py: a python script to read aforementioned pre-trained word-pairs embeddings. 
- DiffVec_Pairs: a text file of word-pairs in DiffVec dataset with gold relaiton labels used to train CGRE_Gold
- DiffVec_Pseudo_Label.txt: a text file of word-pairs in DiffVec dataset with pseudo relaiton labels used to train CGRE_Proxy. The labels are obtained by applying k-mean clustering with k=50. For more details, please refer to the paper. 
- SemEval_Pairs.txt: a text file of word-pairs in SemEval-2012 Task2 test data. 
- Relational Patterns: is a folder for relational patterns of DiffVec training dataset. The folder includes two pickle files as follows:

        1. Patterns_Xmid5Y: a dictionary that maps pattern ids to patterns  
        
        2. Patterns_Xmid5Y_PerPair: a dictionary that maps word-pairs to list of pattern ids. 

        
# Usage
Please follow the process below.

1. download GloVe 300-d from http://nlp.stanford.edu/data/glove.6B.zip, zip the file and put glove.6B.300d.zip into the main folder /Context-Guided-Relation-Embeddings

1. To learn CGRE for SemEval-2012 Task2 word-pairs, run the self-supervised learning of our CGRE model as follows:

    $ cd Context_Guided_RelRep

    $ python train.py

2. To learn NLRA for SemEval-2012 Task2 word-pairs,run the unsupervised NLRA model as follows:

    $ cd NLRA

    $ python train.py

# Cite
If you use the code or the proposed method, please kindly cite the following paper: 

Huda Hakami and Danushka Bollegala: Context-guided Self-Supervised Relation Embeddings Proc. of the 16th International Conference of the Pacific Association for Computational Linguistics (PACLING), October, 2019.
