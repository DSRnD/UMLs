import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.autograd as autograd
#import os



def rank_loss(lossList):
    temp = lossList.argsort()
    return (temp[0][-1])




def createPositiveTripletForLinkPrediction(relationDataset):
    potive_triplet_dict = {}
    for a,  b in relationDataset.items():
        lists= b
        if(len(lists)):
           for c in lists:
                h = c[0]
                t = c[1]
                r = c[-1]
                if(h in potive_triplet_dict.keys()):
                    embed_dict = potive_triplet_dict[h]
                    if(r in embed_dict.keys()):
                        tails =  potive_triplet_dict[h][r]
                        tails.append(t)
                        potive_triplet_dict[h][r] = tails
                    else:
                        potive_triplet_dict[h][r] = [t]
                else:
                    potive_triplet_dict[h] ={}
                    potive_triplet_dict[h][r] =[t]
    return potive_triplet_dict




def transE_Loss( e_s ,e_p , e_o):
        r"""The TransE scoring function.
        .. math::
            f_{TransE}=-||(\mathbf{e}_s + \mathbf{r}_p) - \mathbf{e}_o||_n
        Parameters
        ----------
        e_s : Tensor, shape [n]
            The embeddings of a list of subjects.
        e_p : Tensor, shape [n]
            The embeddings of a list of predicates.
        e_o : Tensor, shape [n]
            The embeddings of a list of objects.
        Returns
        -------
        score : TensorFlow operation
            The operation corresponding to the TransE scoring function.
        

        return tf.negative(
            tf.norm(e_s + e_p - e_o, ord=self.embedding_model_params.get('norm', constants.DEFAULT_NORM_TRANSE),
                    axis=1))"""
        #mat  = e_s + e_p - e_o
        return np.negative(np.linalg.norm((e_s + e_p - e_o), axis = 1))






import random
def Rand(start, end, num, trueTails):
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return  ([i for i in res if i not in trueTails])



def embeddingVectors(model,kg_model_ver):
    _DOC_EMB_FILE = model.d2v._D.detach().cpu().numpy()
    if kg_model_ver == 'transr':
        _REL_EMB_FILE =  model.M_R.detach().cpu().numpy()
        #return _DOC_EMB_FILE, _REL_EMB_FILE
    else:
        _REL_EMB_FILE = model.W_R.detach().cpu().numpy()
        #return _DOC_EMB_FILE, _REL_EMB_FILE
    if kg_model_ver == 'transh':
        _REL_NORM_EMB_FILE =     model.D_R.detach().cpu().numpy()

    print('\nModel embedding loading done!')
    if(kg_model_ver == 'transh'):
        return (_DOC_EMB_FILE, _REL_EMB_FILE, _REL_NORM_EMB_FILE)
    else:
        return (_DOC_EMB_FILE, _REL_EMB_FILE)

def findRanK(head, relation,negT,doc_embedding,rel_embedding):
    es = doc_embedding[head]     #get the embedding of head
    #print(es.shape)
    es = es.reshape(1,es.shape[0])
    es = np.tile(es,(len(negT),1))
    ep = rel_embedding[relation]      #get the embedding of relation
    ep = ep.reshape(1,ep.shape[0])
    ep = np.tile(ep,(len(negT),1))
    #print(es,ep)
    e_t = doc_embedding[negT[0]]
    e_t = e_t.reshape(1,e_t.shape[0])
    #print(e_t)
    del(negT[0])
    negT
    for t in negT:
        et = doc_embedding[t]
        et = et.reshape(1,et.shape[0])
        #print(et)
        e_t = np.row_stack((e_t, et))
        #print(e_t.shape)
    score  = transE_Loss(es ,ep , e_t)
    score = score.reshape(1,score.shape[0])
    return (rank_loss(score)+1)

def findRanKTransr(head, relation,negT,doc_embedding,rel_embedding):
    ep = rel_embedding[relation]
    es = doc_embedding[head]     #get the embedding of head
    ep = torch.from_numpy(ep)
    es = es.reshape(1,es.shape[0])
    es = torch.from_numpy(es)
    print("\nshape of ",ep.shape,es.shape,type(es),type(ep))
    e_s = torch.einsum('ij, ijk -> ik', es, ep)
    print("\nshape of ",ep.shape,es.shape,type(es),type(ep))


import  time
#[t,t1,t2,t3]
def finding_rank(doc_embedding,rel_embedding,potive_triplet_dict,dirnam):
    # [rel_data['h'].iloc[i],rel_data['t'].iloc[i],rel_data['r'].iloc[i]]#Structure of dataset saved in the positive triplet.
    rankList = []
    codeCount  = 50000
    for h,touple in potive_triplet_dict.items():
        for r, tails in touple.items():
            if(r == rel_embedding.shape[0]):
                print("skipped") 
                continue
            for t in tails:
                startTime  = time.time()
                head  = h
                relation  = r
                negativeTriplet = Rand(1, codeCount, len(tails)+5,tails)#will return random indices of code
                negativeTriplet.append(t)
                if('transr' == dirnam):
                    rankList.append(findRanKTransr(head, relation,negativeTriplet,doc_embedding,rel_embedding))
                    #print("--- %s seconds ---" % (time.time() - start_time))
                else:
                    rankList.append(findRanK(head, relation,negativeTriplet,doc_embedding,rel_embedding))
                    #print("--- %s seconds ---" % (time.time() - start_time))
                #print("\nNegative triplet index list",negativeTriplet)
    #print("This is the rank list", rankList)
    return (rankList)

def projection_transH(original, norm):
    return original - np.sum(np.multiply(original ,norm)) * norm

def findRankTansh(head, relation,negT,doc_embedding,rel_embedding,rel_norm_embedding):
    es = doc_embedding[head]     #get the embedding of head
    #print(es.shape)
    es = es.reshape(1,es.shape[0])#h
    #es = np.tile(es,(len(negT),1))
    eNorm = rel_norm_embedding[relation]
    #print(eNorm,eNorm.shape)
    eNorm = eNorm.reshape(1,eNorm.shape[0])
    e_s = projection_transH(es,eNorm)#projected h
    e_s = np.tile(e_s,(len(negT),1))
    ep = rel_embedding[relation]      #get the embedding of relation
    ep = ep.reshape(1,ep.shape[0])
    ep = np.tile(ep,(len(negT),1))
    #print(es,ep)
    et = doc_embedding[negT[0]]
    et = et.reshape(1,et.shape[0])
    e_t = projection_transH(et,eNorm)
    #print(e_t)
    del(negT[0])
    negT
    for t in negT:
        et = doc_embedding[t]
        et = et.reshape(1,et.shape[0])
        et = projection_transH(et,eNorm)
        #print(et)
        e_t = np.row_stack((e_t, et))
        #print(e_t.shape)
    score  = transE_Loss(e_s ,ep , e_t)
    score = score.reshape(1,score.shape[0])
    return (rank_loss(score)+1)	
        
def finding_rank_transH(rel_norm_embedding,doc_embedding,rel_embedding,potive_triplet_dict):
    rankList = []
    codeCount  = 50000
    #print('here is the code count',codeCount)
    for h,touple in potive_triplet_dict.items():
        for r, tails in touple.items():
            if(r == rel_embedding.shape[0]):
                print("skipped")
                continue
            for t in tails:
                start_time  =time.time()
                head  = h
                relation  = r
                negativeTriplet = Rand(1, codeCount, len(tails)+5,tails)#will return random indices of code
                negativeTriplet.append(t)
                #print("\nNegative triplet index list",negativeTriplet)
                rankList.append(findRankTansh(head, relation,negativeTriplet,doc_embedding,rel_embedding,rel_norm_embedding))
                #print("--- %s seconds ---" % (time.time() - start_time))
    print("This is done")
    return (rankList)

def mrr_score(ranks):
    r"""Mean Reciprocal Rank (MRR)
    The function computes the mean of the reciprocal of elements of a vector of rankings ``ranks``.
    It is used in conjunction with the learning to rank evaluation protocol of
    :meth:`ampligraph.evaluation.evaluate_performance`.
    It is formally defined as follows:
    .. math::
        MRR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}\frac{1}{rank_{(s, p, o)_i}}
    where :math:`Q` is a set of triples and :math:`(s, p, o)` is a triple :math:`\in Q`.
    .. note::
        This metric is similar to mean rank (MR) :meth:`ampligraph.evaluation.mr_score`. Instead of averaging ranks,
        it averages their reciprocals. This is done to obtain a metric which is more robust to outliers.
    Consider the following example. Each of the two positive triples identified by ``*`` are ranked
    against four corruptions each. When scored by an embedding model, the first triple ranks 2nd, and the other triple
    ranks first. The resulting MRR is: ::
        s	 p	   o		score	rank
        Jack   born_in   Ireland	0.789	   1
        Jack   born_in   Italy		0.753	   2  *
        Jack   born_in   Germany	0.695	   3
        Jack   born_in   China		0.456	   4
        Jack   born_in   Thomas		0.234	   5
        s	 p	   o		score	rank
        Jack   friend_with   Thomas	0.901	   1  *
        Jack   friend_with   China      0.345	   2
        Jack   friend_with   Italy      0.293	   3
        Jack   friend_with   Ireland	0.201	   4
        Jack   friend_with   Germany    0.156	   5
        MRR=0.75
    Parameters
    ----------
    ranks: ndarray or list, shape [n] or [n,2]
        Input ranks of n test statements.
    Returns
    -------
    mrr_score: float
        The MRR score
    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.evaluation.metrics import mrr_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> mrr_score(rankings)
    0.4375
    """
    if isinstance(ranks, list):
        ranks = np.asarray(ranks)
    #print(type(ranks),ranks)
    ranks = ranks.reshape(-1)
    return np.sum(1 / ranks) / len(ranks)

def mr_score(ranks):
    r"""Mean Rank (MR)
    The function computes the mean of of a vector of rankings ``ranks``.
    It can be used in conjunction with the learning to rank evaluation protocol of
    :meth:`ampligraph.evaluation.evaluate_performance`.
    It is formally defined as follows:
    .. math::
        MR = \frac{1}{|Q|}\sum_{i = 1}^{|Q|}rank_{(s, p, o)_i}
    where :math:`Q` is a set of triples and :math:`(s, p, o)` is a triple :math:`\in Q`.
    .. note::
        This metric is not robust to outliers.
        It is usually presented along the more reliable MRR :meth:`ampligraph.evaluation.mrr_score`.
    Consider the following example. Each of the two positive triples identified by ``*`` are ranked
    against four corruptions each. When scored by an embedding model, the first triple ranks 2nd, and the other triple
    ranks first. The resulting MR is: ::
        s	 p	   o		score	rank
        Jack   born_in   Ireland	0.789	   1
        Jack   born_in   Italy		0.753	   2  *
        Jack   born_in   Germany	0.695	   3
        Jack   born_in   China		0.456	   4
        Jack   born_in   Thomas		0.234	   5
        s	 p	   o		score	rank
        Jack   friend_with   Thomas	0.901	   1  *
        Jack   friend_with   China      0.345	   2
        Jack   friend_with   Italy      0.293	   3
        Jack   friend_with   Ireland	0.201	   4
        Jack   friend_with   Germany    0.156	   5
        MR=1.5
    Parameters
    ----------
    ranks: ndarray or list, shape [n] or [n,2]
        Input ranks of n test statements.
    Returns
    -------
    mr_score: float
        The MR score
    Examples
    --------
    >>> from ampligraph.evaluation import mr_score
    >>> ranks= [5, 3, 4, 10, 1]
    >>> mr_score(ranks)
    4.6
    """
    if isinstance(ranks, list):
        ranks = np.asarray(ranks)
    ranks = ranks.reshape(-1)
    return np.sum(ranks) / len(ranks)
	
def hits_at_n_score(ranks, n):
    r"""Hits@N
    The function computes how many elements of a vector of rankings ``ranks`` make it to the top ``n`` positions.
    It can be used in conjunction with the learning to rank evaluation protocol of
    :meth:`ampligraph.evaluation.evaluate_performance`.
    It is formally defined as follows:
    .. math::
        Hits@N = \sum_{i = 1}^{|Q|} 1 \, \text{if } rank_{(s, p, o)_i} \leq N
    where :math:`Q` is a set of triples and :math:`(s, p, o)` is a triple :math:`\in Q`.
    Consider the following example. Each of the two positive triples identified by ``*`` are ranked
    against four corruptions each. When scored by an embedding model, the first triple ranks 2nd, and the other triple
    ranks first. Hits@1 and Hits@3 are: ::
        s	 p	   o		score	rank
        Jack   born_in   Ireland	0.789	   1
        Jack   born_in   Italy		0.753	   2  *
        Jack   born_in   Germany	0.695	   3
        Jack   born_in   China		0.456	   4
        Jack   born_in   Thomas		0.234	   5
        s	 p	   o		score	rank
        Jack   friend_with   Thomas	0.901	   1  *
        Jack   friend_with   China      0.345	   2
        Jack   friend_with   Italy      0.293	   3
        Jack   friend_with   Ireland	0.201	   4
        Jack   friend_with   Germany    0.156	   5
        Hits@3=1.0
        Hits@1=0.5
    Parameters
    ----------
    ranks: ndarray or list, shape [n] or [n,2]
        Input ranks of n test statements.
    n: int
        The maximum rank considered to accept a positive.
    Returns
    -------
    hits_n_score: float
        The Hits@n score
    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.evaluation.metrics import hits_at_n_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> hits_at_n_score(rankings, n=3)
    0.5
    """

    if isinstance(ranks, list):
        ranks = np.asarray(ranks)
    ranks = ranks.reshape(-1)
    return np.sum(ranks <= n) / len(ranks)
	
	

def CreateMetricsForLinkPrediction(potive_triplet_dict,kg_model_ver,epoch_i,model,output_dir):
    fileName = '/LinkPredictionResult'+str(epoch_i+1)+kg_model_ver+'.txt'
    filepath = (output_dir+fileName)
    with open(filepath, 'w+') as w:
        w.write("Method\tmrrScore\tmrScore\thitsat_1_score\thitsat_3_score,hitsat_10_score")
        if(kg_model_ver  == 'dbow' or kg_model_ver =='distmult' or kg_model_ver == 'dm' or kg_model_ver =='transd' or kg_model_ver =='transd_dbow' or kg_model_ver =='transd_dm' or kg_model_ver == 'transe'):
            doc_embedding, rel_embedding = embeddingVectors(model,kg_model_ver)
            rankList = finding_rank(doc_embedding,rel_embedding,potive_triplet_dict,kg_model_ver)
        elif (kg_model_ver =='transh'):
            doc_embedding, rel_embedding, rel_norm_embedding = embeddingVectors(model,kg_model_ver)
            rankList = finding_rank_transH(rel_norm_embedding,doc_embedding,rel_embedding,potive_triplet_dict)
        mrrScore = mrr_score(rankList)
        mrScore = mr_score(rankList)
        hitsat_1_score = hits_at_n_score(rankList,1)
        hitsat_3_score = hits_at_n_score(rankList,3)
        hitsat_10_score = hits_at_n_score(rankList,10)
        w.write("\n%s\t%s\t%s\t%s\t%s\t%s"%(kg_model_ver,mrrScore,mrScore,hitsat_1_score,hitsat_3_score,hitsat_10_score))
        print('Metrics generated for validation dataset for LinkPrediction task of epoch', epoch_i)

