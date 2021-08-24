import argparse
import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.nn import functional as F
import numpy as np

import random
def Rand(start, end, num, trueTails):
    negative = []
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    for i in res:
        if(i not in trueTails):
            negative.append(i)
            if(len(trueTails) == len(negative)):
                return negative


def generateNegativeTriplet(potive_triplet_dict):
    pos_neg_triplet ={}
    tripletCount = 0 
    for r,touple in potive_triplet_dict.items():
        for h, tails in touple.items():
            if(r == 304):
                tripletCount += len(tails)
                print("skipped") 
                continue
            negative = Rand(0, 757490,2*len(tails),tails)#Upper limit has to be given
            pos_neg_triplet[r] = {}
            pos_neg_triplet[r][h] = []
            pos_neg_triplet[r][h] = [tails,negative]
    print("\Total code skipped", tripletCount)
    return pos_neg_triplet

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



def createPositiveTriplet(relationDataset):
    potive_triplet_dict = {}
    for lists in relationDataset:
        if(len(lists)):
            for list in lists:
                h = list[0]
                t = list[1]
                r = list[-1]
                if(r in potive_triplet_dict.keys()):
                    embed_dict = potive_triplet_dict[r]
                    if(h in embed_dict.keys()):
                        tails =  potive_triplet_dict[r][h]
                        tails.append(t)
                        potive_triplet_dict[r][h] = tails
                    else:
                        potive_triplet_dict[r][h] = [t]
                else:
                    potive_triplet_dict[r] ={}
                    potive_triplet_dict[r][h] =[t]
    print(len(potive_triplet_dict))
    return generateNegativeTriplet(potive_triplet_dict)


def load_rel_dataset(filepath):
    potive_triplet_dict ={}
    df = pd.read_csv(filepath, sep='\t',header=None)
    print('dataframe',df.shape)
    positive_triplet = df[1000001:1200000]#Assuming that triplet are in the format 2nd colm:- h, 3rd col:- t and 4th col:- r
    print('triplet count',positive_triplet.shape,positive_triplet.dtypes )
    positive_triplet.columns =['medicalcode','h','t','r']
    
    for index,row in  positive_triplet.iterrows():
        if(index%10000 ==0):
            print(index)
        h = row['h']
        r = row['r']
        t = row['t']
        print('triplet',h,r,t)
        if(r in potive_triplet_dict.keys()):
            embed_dict = potive_triplet_dict[r]
            if(h in embed_dict.keys()):
                potive_triplet_dict[r][h].append(t)
            else:
                potive_triplet_dict[r][h] = []
                potive_triplet_dict[r][h].append(t)
        else:
            potive_triplet_dict[r] ={}
            potive_triplet_dict[r][h] =[]
            potive_triplet_dict[r][h].append(t)
    print(len(potive_triplet_dict))
    return generateNegativeTriplet(potive_triplet_dict)


def score_emb_distmult(s_emb,p_emb,o_emb):
        #n = p_emb.size(0)
        out = (s_emb * p_emb * o_emb).sum(dim=1)
        #print("\here is the loss value",out,out.shape)
        return out.item()
        
def score_emb_transe(s_emb, p_emb, o_emb):#return a single value
        out = -F.pairwise_distance(s_emb + p_emb, o_emb, p= 2 )
        return out.item()        


def score_spo(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model):
        r"""Compute scores for a set of triples.
        `s`, `p`, and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.
        `direction` may influence how scores are computed. For most models, this setting
        has no meaning. For reciprocal relations, direction must be either `"s"` or
        `"o"` (depending on what is predicted).
        Returns a vector of size :math:`n`, in which the :math:`i`-th entry holds the
        score of triple :math:`(s_i, p_i, o_i)`.
        """
        if(model in ['transe','dbow','dm','transd_dbow','transd','trans_dm']):
            s = doc_embedding[head]
            s = (torch.tensor(s)).view(1,-1)
            p = rel_embedding[relation]
            p = (torch.tensor(p)).view(1,-1)
            oPos = doc_embedding[tail_pos]
            oPos = (torch.tensor(oPos)).view(1,-1)
            oNeg = doc_embedding[tail_neg]
            oNeg = (torch.tensor(oNeg)).view(1,-1)#Convert  numpy array into tensor
            #print("shape of tensor\n",s.shape,p.shape,oPos.shape,oNeg.shape)
            posScore  = score_emb_transe(s, p, oPos)
            negScore  = score_emb_transe(s, p, oNeg)
            return posScore, negScore
        if(model in ['distmult']):
            s = doc_embedding[head]
            s = (torch.tensor(s)).view(1,-1)
            p = rel_embedding[relation]
            p = (torch.tensor(p)).view(1,-1)
            oPos = doc_embedding[tail_pos]
            oPos = (torch.tensor(oPos)).view(1,-1)
            oNeg = doc_embedding[tail_neg]
            oNeg = (torch.tensor(oNeg)).view(1,-1)#Convert  numpy array into tensor
            #print("shape of tensor\n",s.shape,p.shape,oPos.shape,oNeg.shape)
            posScore  = score_emb_distmult(s, p, oPos)
            negScore  = score_emb_distmult(s, p, oNeg)
            return posScore, negScore
        #return self._scorer.score_emb(s, p, o, combine="spo").view(-1)#This would return a tensor of torch.Size([16])	

def NegativeTail(start, end, num, trueTails):#Returning a single negative tail
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    for i in res:
        if not( i == trueTails):
            return i


		
def valGet_X_y(model, post_neg_triplet,doc_embedding,rel_embedding):
    """
    :param model: kge.model.KgeModel
    :param pos_spo: torch.Tensor of positive triples
    :param neg_spo: torch.Tensor of negative triples
    :return X: torch.Tensor of [pos_scores, neg_scores]
    :return y: torch.Tensor of [1s, 0s]
    """
    totalCount  = 0
    relationDict_score  = {}
    pos_scores = []
    neg_scores = []
    relList = []
    relationDict = {}
    for lists in post_neg_triplet:
        if(len(lists)):
            for c in lists:
                head = c[0]
                tail_pos = c[1]
                relation = c[-1]
                if(relation == 304):
                    continue
                tail_neg = NegativeTail(0, 50000, 2, tail_pos)
                pos_score, neg_score = score_spo(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model)
                pos_scores.append(pos_score)
                neg_scores.append(pos_score)
                relList.append(relation)
                if( relation in relationDict_score.keys()):
                    posScore = relationDict_score[relation][0]
                    negScore = relationDict_score[relation][1]
                    posScore.append(pos_score)
                    negScore.append(neg_score)
                    relationDict_score[relation] = [posScore, negScore]
                else:
                    relationDict_score[relation] = [[pos_score],[neg_score]]
    for key, value in relationDict_score.items():
        tripletCount = len(value[0])
        totalCount += tripletCount
        poScore = torch.FloatTensor(value[0])
        neScore = torch.FloatTensor(value[1])
        X_r = torch.reshape(torch.cat((poScore, neScore)), (-1, 1))
        """y_r = torch.cat(
        (
            torch.ones_like(poScore, device="gpu"),
            torch.zeros_like(neScore, device="gpu"),
        )
        )"""
        #threshold = get_threshold(X_r,y_r)
        threshold = torch.min(poScore).item()
        relationDict[key] = threshold
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    print("\nConfirming Total Count",totalCount)
    return dict_score_rel

    '''for index,row in  df.iterrows():
        i += 1
        if(i%10000 ==0):
            print(index)
        head = row['h']
        relation = row['r']
        tail_pos = row['t']
        if(relation == 304):
            continue
        tail_neg = NegativeTail(0, 757490, 2, tail_pos)
        pos_score, neg_score = score_spo(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model)
        pos_scores.append(pos_score)
        neg_scores.append(pos_score)
        relList.append(relation)
        if( relation in relationDict_score.keys()):
            posScore = relationDict_score[relation][0]
            negScore = relationDict_score[relation][1]
            posScore.append(pos_score)
            negScore.append(neg_score)
            relationDict_score[relation] = [posScore, negScore]
        else:
            relationDict_score[relation] = [[pos_score],[neg_score]]
    for key, value in relationDict_score.items():
        tripletCount = len(value[0])
        totalCount += tripletCount
        poScore = torch.FloatTensor(value[0])
        neScore = torch.FloatTensor(value[1])
        X_r = torch.reshape(torch.cat((poScore, neScore)), (-1, 1))
        """y_r = torch.cat(
        (
            torch.ones_like(poScore, device="gpu"),
            torch.zeros_like(neScore, device="gpu"),
        )
        )"""
        #threshold = get_threshold(X_r,y_r)
        threshold = torch.min(poScore).item()
        relationDict[key] = threshold
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    print("\nConfirming Total Count",totalCount)
    return dict_score_rel'''


def projection_transH(original, norm):
    return original - torch.sum(torch.multiply(original ,norm)) * norm


def score_spo_transH(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model,rel_norm_embedding):
        r"""Compute scores for a set of triples.
        `s`, `p`, and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.
        `direction` may influence how scores are computed. For most models, this setting
        has no meaning. For reciprocal relations, direction must be either `"s"` or
        `"o"` (depending on what is predicted).
        Returns a vector of size :math:`n`, in which the :math:`i`-th entry holds the
        score of triple :math:`(s_i, p_i, o_i)`.
        """
        if(model in ['transh','dbow_transh','dm_transh']):
            s = doc_embedding[head]
            s = (torch.tensor(s)).view(1,-1)
            p = rel_embedding[relation]
            p = (torch.tensor(p)).view(1,-1)
            oPos = doc_embedding[tail_pos]
            oPos = (torch.tensor(oPos)).view(1,-1)
            oNeg = doc_embedding[tail_neg]
            oNeg = (torch.tensor(oNeg)).view(1,-1)#Convert  numpy array into tensor
            #print("shape of tensor\n",s.shape,p.shape,oPos.shape,oNeg.shape)
            norm  = rel_norm_embedding[relation]
            norm = (torch.tensor(norm)).view(1,-1)
            s_0 = projection_transH(s,norm)
            oPos_0 = projection_transH(oPos,norm)
            oNeg_0 = projection_transH(oNeg,norm)
            posScore  = score_emb_transe(s_0, p, oPos_0)
            negScore  = score_emb_transe(s_0, p, oNeg_0)
            return posScore, negScore   
    
    
def valGet_X_y_transH(model, post_neg_triplet,doc_embedding,rel_embedding,rel_norm_embedding):
    totalCount  = 0
    relationDict_score  = {}
    pos_scores = []
    neg_scores = []
    relList = []
    relationDict = {}
    #for triplet in post_neg_triplet:
    #df = pd.read_csv(post_neg_triplet,sep='\t',header=None)
    #df = df[:900000]
    #df.columns =['medicalcode','h','t','r']
    #df.drop_duplicates()
    i = 0
    for lists in post_neg_triplet:
        if(len(lists)):
            for c in lists:
                head = c[0]
                tail_pos = c[1]
                relation = c[-1]
                if(relation == 304):
                    continue
                tail_neg = NegativeTail(0, 50000, 2, tail_pos)
                pos_score, neg_score = score_spo_transH(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model, rel_norm_embedding)
                pos_scores.append(pos_score)
                neg_scores.append(pos_score)
                relList.append(relation)
                if( relation in relationDict_score.keys()):
                    posScore = relationDict_score[relation][0]
                    negScore = relationDict_score[relation][1]
                    posScore.append(pos_score)
                    negScore.append(neg_score)
                    relationDict_score[relation] = [posScore, negScore]
                else:
                    relationDict_score[relation] = [[pos_score],[neg_score]]
    for key, value in relationDict_score.items():
        print("\nProcessing relation", key )
        tripletCount = len(value[0])
        totalCount += tripletCount
        poScore = torch.FloatTensor(value[0])
        neScore = torch.FloatTensor(value[1])
        X_r = torch.reshape(torch.cat((poScore, neScore)), (-1, 1))
        threshold = torch.min(poScore).item()
        relationDict[key] = threshold
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    #print("\nConfirming Total Count", totalCount)
    return dict_score_rel

    '''for index,row in  df.iterrows():
        i += 1
        if(i%10000 ==0):
            print(index)
        head = row['h']
        relation = row['r']
        tail_pos = row['t']
        if(relation == 304):
            continue
        tail_neg = NegativeTail(0, 757490, 2, tail_pos)
        pos_score, neg_score = score_spo_transH(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model, rel_norm_embedding)
        pos_scores.append(pos_score)
        neg_scores.append(pos_score)
        relList.append(relation)
        if( relation in relationDict_score.keys()):
            posScore = relationDict_score[relation][0]
            negScore = relationDict_score[relation][1]
            posScore.append(pos_score)
            negScore.append(neg_score)
            relationDict_score[relation] = [posScore, negScore]
        else:
            relationDict_score[relation] = [[pos_score],[neg_score]]
    for key, value in relationDict_score.items():
        print("\nProcessing relation", key )
        tripletCount = len(value[0])
        totalCount += tripletCount
        poScore = torch.FloatTensor(value[0])
        neScore = torch.FloatTensor(value[1])
        
        X_r = torch.reshape(torch.cat((poScore, neScore)), (-1, 1))
        threshold = torch.min(poScore).item()
        relationDict[key] = threshold
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    print("\nConfirming Total Count", totalCount)
    return dict_score_rel'''

def testGet_X_y(model, post_neg_triplet,doc_embedding,rel_embedding):
    """
    :param model: kge.model.KgeModel
    :param pos_spo: torch.Tensor of positive triples
    :param neg_spo: torch.Tensor of negative triples
    :return X: torch.Tensor of [pos_scores, neg_scores]
    :return y: torch.Tensor of [1s, 0s]
    """
    totalCount  = 0
    relationDict_score  = {}
    pos_scores = []
    neg_scores = []
    relList = []
    relationDict = {}
    #df = pd.read_csv(post_neg_triplet,sep='\t',header=None)
    #df = df[900000:1200000]
    #df.columns =['medicalcode','h','t','r']
    #df.drop_duplicates()
    i = 0
    for lists in post_neg_triplet:
        if(len(lists)):
            for c in lists:
                head = c[0]
                tail_pos = c[1]
                relation = c[-1]
                if(relation == 304):
                    continue
                tail_neg = NegativeTail(0, 50000, 2, tail_pos)
                pos_score, neg_score = score_spo(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model)
                pos_scores.append(pos_score)
                neg_scores.append(pos_score)
                relList.append(relation)
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    return dict_score_rel

    '''for index,row in  df.iterrows():
        i += 1
        if(i%10000 ==0):
            print(index)
        head = row['h']
        relation = row['r']
        tail_pos = row['t']
        if(relation == 304):
            continue
        tail_neg = NegativeTail(0, 757490, 2, tail_pos)
        pos_score, neg_score = score_spo(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model)
        pos_scores.append(pos_score)
        neg_scores.append(pos_score)
        relList.append(relation)
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    return dict_score_rel'''
	
def testGet_X_y_transH(model, post_neg_triplet,doc_embedding,rel_embedding,rel_norm_embedding):
    totalCount  = 0
    relationDict_score  = {}
    pos_scores = []
    neg_scores = []
    relList = []
    relationDict = {}
    #for triplet in post_neg_triplet:
    #df = pd.read_csv(post_neg_triplet,sep='\t',header=None)
    #df = df[900000:1200000]
    #df.columns =['medicalcode','h','t','r']
    #df.drop_duplicates()
    i = 0
    #for index,row in  df.iterrows():y
    for lists in post_neg_triplet:
        if(len(lists)):
            for c in lists:
                head = c[0]
                tail_pos = c[1]
                relation = c[-1]
                if(relation == 304):
                    continue
                tail_neg = NegativeTail(0, 50000, 2, tail_pos)
                pos_score, neg_score = score_spo_transH(head, relation, tail_pos,tail_neg, doc_embedding,rel_embedding, model, rel_norm_embedding)
                pos_scores.append(pos_score)
                neg_scores.append(pos_score)
                relList.append(relation)
    pos_scores = torch.FloatTensor(pos_scores)
    neg_scores = torch.FloatTensor(neg_scores)
    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )
    dict_score_rel = {'X_0' : X, 'y_0':y ,'relList_0' : relList, 'relationDict_0' : relationDict}
    return dict_score_rel

def CreateMetricsTriplet(relationDataset,kg_model_ver,epoch_i,model,output_dir):
    fileName = '/TripletClassificationResult'+str(epoch_i+1)+kg_model_ver+'.txt'
    filepath = (output_dir+fileName)
    with open(filepath, 'w+') as w:
        w.write('Method\tAccuracy\tF1Score\n')
        if(kg_model_ver  == 'dbow' or kg_model_ver =='distmult' or kg_model_ver == 'dm' or kg_model_ver =='transd' or kg_model_ver =='transd_dbow' or kg_model_ver =='transd_dm' or kg_model_ver == 'transe'):
            doc_embedding, rel_embedding = embeddingVectors(model,kg_model_ver)
            score_rel_dict = valGet_X_y(kg_model_ver,relationDataset[:200000], doc_embedding,rel_embedding)
            X_valid = score_rel_dict['X_0']
            #y_valid = score_rel_dict['y_0']
            #relList =  score_rel_dict['relList_0']
            threshold_dict = score_rel_dict['relationDict_0']
            #valid_relations = list(set(relList))
            REL_KEY = -1
            threshold_dict[REL_KEY] = torch.min(X_valid).item()
            score_rel_dict = testGet_X_y(kg_model_ver,relationDataset[200000:len(relationDataset)], doc_embedding,rel_embedding)#Here get the value of test data
            X_valid = score_rel_dict['X_0']
            y_valid = score_rel_dict['y_0']
            relList =  score_rel_dict['relList_0']
            #threshold_dict = score_rel_dict['relationDict_0']
            valid_relations = list(set(relList))
            y_test = y_valid.numpy()#Will be used for cal
            y_pred = []
                #y_pred_valid = torch.zeros(y_valid.shape, dtype=torch.long, device="cpu")#shape  of [[n]]
                #y_pred_test = torch.zeros(y_test.shape, dtype=torch.long, device="cpu")#shape  of [[n]]
            valid_spo_all = relList * 2
            for i in range(X_valid.shape[0]):
                rel = valid_spo_all[i]
                if(rel in threshold_dict.keys() ):
                    if((X_valid[i][0]).item()>= threshold_dict[rel]):
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
                else:
                    if((X_valid[i][0]).item()>= threshold_dict[-1]):
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            y_pred = np.array(y_pred)
            test_accuracy=accuracy_score(y_test, y_pred)
            test_f1=f1_score(y_test, y_pred)
            model_file=model
            w.write("\n%s\t%s\t%s"%(kg_model_ver,test_accuracy,test_f1))
        elif (kg_model_ver =='transh'):
            doc_embedding, rel_embedding, rel_norm_embedding = embeddingVectors(model, kg_model_ver)
            score_rel_dict = valGet_X_y_transH(kg_model_ver,relationDataset[:200000], doc_embedding,rel_embedding,rel_norm_embedding)
            X_valid = score_rel_dict['X_0']
            threshold_dict = score_rel_dict['relationDict_0'] 
            REL_KEY = -1
            threshold_dict[REL_KEY] = torch.min(X_valid).item()
            print('\nThreshold generation of relation has been done now')
            score_rel_dict = testGet_X_y_transH(kg_model_ver,relationDataset[200000:len(relationDataset)], doc_embedding,rel_embedding,rel_norm_embedding)
            X_valid = score_rel_dict['X_0']
            y_valid = score_rel_dict['y_0']
            relList =  score_rel_dict['relList_0']
            valid_relations = list(set(relList))
            y_test = y_valid.numpy()#Will be used for cal
            y_pred = []
            valid_spo_all = relList * 2
            for i in range(X_valid.shape[0]):
                    rel = valid_spo_all[i]
                    if(rel in threshold_dict.keys() ):
                        if((X_valid[i][0]).item()>= threshold_dict[rel]):
                            y_pred.append(1)
                        else:
                            y_pred.append(0)
                    else:
                        if((X_valid[i][0]).item()>= threshold_dict[-1]):
                            y_pred.append(1)
                        else:
                            y_pred.append(0)
            y_pred = np.array(y_pred)
            test_accuracy=accuracy_score(y_test, y_pred)
            test_f1=f1_score(y_test, y_pred)
            model_file=model
            w.write("\n%s\t%s\t%s"%(kg_model_ver,test_accuracy,test_f1))
    print('\nTriplet Classificaiton task done for epoch',epoch_i+1)
    