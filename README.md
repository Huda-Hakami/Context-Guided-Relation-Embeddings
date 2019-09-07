# CGRE: Context-Guided-Relation-Embeddings
This directory contains datasets and implementations of context-guided relation embedding (CGRE) proposed in our paper: Context-guided Self-Supervised Relation Embeddings.

It also implement neural latent relational analysis (NLRA) for word-pair representations, a method proposed by Washio and Kato, 2018 in their paper: Neural Latent Relational Analysis to Capture Lexical Semantic Relations in a Vector Space.

We used python 3.6.1.
These codes require tensorflow, sklearn and numpy.
This software includes the work that is distributed in the Apache License 2.0.
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
