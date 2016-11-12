from __future__ import division
from sklearn.preprocessing import normalize
import numpy as np
import random

def stochasticCVB0(documents=None,
                   numTopics=None,
                   numWords=None,
                   numIterations=None, 
                   wordDict=None, 
                   burnInPerDoc=1, 
                   minibatchSize=10, 
                   alpha=0.1, 
                   eta=0.1,  
                   tau=5, 
                   kappa=0.9, 
                   tau2=5, 
                   kappa2=0.9, 
                   scale=1):
    
    #Stochastic CVB0 for LDA. 
    #Reference: J. R. Foulds, L. Boyles, C. DuBois, P. Smyth and M. Welling.
    #           Stochastic collapsed variational Bayesian inference 
    #           for latent Dirichlet allocation. Proceedings of the 
    #           19th ACM SIGKDD International Conference on 
    #           Knowledge Discovery and Data Mining (KDD), 2013. 
    #
    #@param documents is an numpy array containing the word indices for each token.
    #@param numWords, numTopics. size of topic matrix.
    #@param numIterations number of iterations.
    #@param wordDict a containers. Map object that maps word indices to words.
    #@param burnInPerDoc number of burn-in passes per doc to learn theta.
    #@param minibatchSize number of documents in minibatch.  Set to one to not use minibatches.
    #@param alpha, eta. Dirichlet prior concentration parameters.
    #@param saveParams a boolean determining whether per-iteration results are stored in VBparamsPerIter.
    #@param saveEvery if saveParams, how many document iterations to wait between recorded results.
    #recorded in VBparamsPerIter, as well as the final result in VBparams.
    #@param tau, kappa, scale. Step size parameters for topics.
    #@param tau2, kappa2. Step size parameters for theta.
    
    numDocuments = len(documents);
    alpha        = np.ones(numTopics) * alpha
    eta          = np.ones(numWords)  * eta
    etaSum       = sum(eta)
    
    randomDocId     = random.randint(0,numDocuments-1)
    numTokens       = numDocuments * len(documents[randomDocId])
    topicCounts     = np.ones(numTopics) * numTokens / numTopics
    wordTopicCounts = np.ones((numWords, numTopics)) * numTokens / (numWords * numTopics)
    
    documentsLength    = (np.vectorize(lambda x: x.size)(documents)).reshape(numDocuments,1)
    totalWordsInCorpus = documentsLength.sum()
    
    documentTopicCounts = np.random.rand(numDocuments, numTopics)
    documentTopicCounts = normalize(documentTopicCounts, axis=1, norm="l1") * documentsLength
    
    noOfBatches            = numDocuments // minibatchSize
    batchWordTopicCounts   = np.zeros((numWords, numTopics))
    batchTopicCounts       = np.zeros(numTopics)
    miniBatchWordsinCorpus = 0
    
    stepSize = scale / (tau**kappa)
    for i in range(numIterations):
        docId = i % numDocuments
        docLength = documentsLength[docId]
        
        for burn in range(burnInPerDoc):
            for i,wordId in enumerate(documents[docId]):
                stepSize2 = (i + burn * docLength + tau2) ** -kappa2
                
                # SCVB0 estimate of gamma for current token.
                probs = (wordTopicCounts[wordId] + eta[wordId]) \
                         *(documentTopicCounts[docId] + alpha) \
                         / (topicCounts + etaSum)

                probs = probs / sum(probs)
                
                # update document statistics.
                documentTopicCounts[docId] = (1 - stepSize2) * documentTopicCounts[docId] + \
                    stepSize2 * docLength * probs

        # Process a document
        for i,wordId in enumerate(documents[docId]):
            stepSize2 = (i + burnInPerDoc * docLength + tau2) ** -kappa2

            # SCVB0 estimate of gamma for current token.
            probs = (wordTopicCounts[wordId] + eta[wordId]) \
                 *(documentTopicCounts[docId] + alpha) \
                 /(topicCounts + etaSum)

            probs = probs / sum(probs);

            documentTopicCounts[docId] = (1 - stepSize2) * documentTopicCounts[docId] + \
                stepSize2 * docLength * probs

            # update minibatch counts for topic statistics
            batchWordTopicCounts[wordId] = batchWordTopicCounts[wordId] + probs
            batchTopicCounts = batchTopicCounts + probs

            miniBatchWordsinCorpus += 1

        # Processed a batch
        if i % minibatchSize == 0:
            stepSize = 1 - (1-stepSize)**minibatchSize

            # Update the minibatch results.
            wordTopicCounts = (1 - stepSize) * wordTopicCounts + \
                stepSize * (totalWordsInCorpus/miniBatchWordsinCorpus) * batchWordTopicCounts
            topicCounts = (1 - stepSize) * topicCounts + \
                stepSize * (totalWordsInCorpus/miniBatchWordsinCorpus) * batchTopicCounts
            
            batchWordTopicCounts = batchWordTopicCounts * 0
            batchTopicCounts = batchTopicCounts * 0
            miniBatchWordsinCorpus = 0

            stepSize = scale / ((i + tau)**kappa)
    
    print ("Top Words for each topic")
    for topicWordCounts in wordTopicCounts.T:
        print([wordDict[w] for w in np.array(topicWordCounts).argsort()[-10:]])
        
        
import scipy.io as sio
data = sio.loadmat('nips.mat')
documents = [d[0][0].flatten() -1 
             for d in data["documents"]]
numWords = data["numWords"][0][0]
keys = sio.loadmat('keys.mat')
word_ids = [k[0][0]-1 
            for k in keys["Key"][0]]
vals = sio.loadmat('vals.mat')
words = [v[0] 
         for v in vals["Vals"][0]]
wordDict = dict(zip(word_ids, words))

stochasticCVB0(documents=documents, 
               numTopics=50, 
               numWords=numWords, 
               numIterations=30000, 
               wordDict=wordDict)
