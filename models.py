import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.autograd as autograd

class NegativeSampling(nn.Module):
    """Negative sampling loss as proposed by T. Mikolov et al. in Distributed
    Representations of Words and Phrases and their Compositionality.
    """
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        """Computes the value of the loss function.

        Parameters
        ----------
        scores: autograd.Variable of size (batch_size, num_noise_words + 1)
            Sparse unnormalized log probabilities. The first element in each
            row is the ground truth score (i.e. the target), other elements
            are scores of samples from the noise distribution.
        """
        try:
            k = scores.size()[1] - 1
            
            return -torch.sum(
            self._log_sigmoid(scores[:, 0])
            + torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k
        ) / scores.size()[0]
        
        except:
            k = 1
            
            return -torch.sum(torch.sum(self._log_sigmoid(scores)))

class marginLoss(nn.Module):
	def __init__(self):
		super(marginLoss, self).__init__()

	def forward(self, pos, neg, margin):
		val = pos - neg + margin
		return torch.sum(torch.max(val, torch.zeros_like(val)))
        
def projection_transH(original, norm):
    return original - torch.sum(original * norm, dim=1, keepdim=True) * norm
    
def projection_DistMult(original, norm1, norm2):
    return torch.sum(original * norm1, dim=1, keepdim=True) * norm2
    
def projection_transD(original, norm):
    return original + torch.sum(original * norm, dim=1, keepdim=True) * norm
    
class Link_Model(nn.Module):
    """Link prediction model"""
    
    def __init__(self, kg_model):
        self.vec_dim = vec_dinm
        self.out_dim = out_dim
        self.kg_model = kg_model
        #self.linear1 = nn.Linear(self.vec_dim, self.out_dim)
        self.linear = nn.Linear(1, 1, bias=True, requires_grad=True)
        
    def forward(self, hi, ti, ri):
        with torch.no_grad():
            _, _, _, _, pos, _ = self.kg_model(_, _, _, hi, ti, ri, _)
        
        out = self.linear(pos)
        
        return out
        
class D2V_KG(nn.Module):
    """Doc2vec model with transh loss
    """
    def __init__(self, vec_dim, num_docs, num_words, n_rel, d2v_model_ver, kg_model_ver, margin, delta):
        super(D2V_KG, self).__init__()
        self.num_docs = num_docs
        self.margin = margin
        self.delta = delta
        
        self.kg_model_ver = kg_model_ver
        self.d2v_model_ver = d2v_model_ver
        
        if d2v_model_ver == 'dm':
            self.d2v = DM(vec_dim, num_docs, num_words)
        else:
            self.d2v = DBOW(vec_dim, num_docs, num_words)
            
        self.cost_func = NegativeSampling()
        self.kg_loss_fn = marginLoss()
        
        self.W_R = nn.Parameter(
            torch.randn(n_rel, vec_dim), requires_grad=True)
        self.D_R = nn.Parameter(
            torch.randn(n_rel, vec_dim), requires_grad=True)
        self.M_R = nn.Parameter(
            torch.randn(n_rel, vec_dim, vec_dim), requires_grad=True)
            
        normalize_entity_emb = F.normalize(self.d2v._D.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.W_R.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.D_R.data, p=2, dim=1)
        
        self.d2v._D.data = normalize_entity_emb
        self.W_R.data = normalize_relation_emb
        self.D_R.data = normalize_norm_emb
        
    def forward(self, context_ids, doc_ids, target_noise_ids, hi, ti, ri, tj):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        context_ids: torch.Tensor of size (batch_size, num_context_words)
            Vocabulary indices of context words.

        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.
            
        hi: torch.Tensor of size (batch_size,)
            Heads from golden triplets from relational graph
            
        ti: torch.Tensor of size (batch_size,)
            Tails from golden triplets from relational graph
        
        ri: torch.Tensor of size (batch_size,)
            Relations from golden triplets from relational graph
            
        tj: torch.Tensor of size (batch_size,)
            Tails from noisy triplets from relational graph

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        
        hi_emb = self.d2v._D[hi,:]
        ti_emb = self.d2v._D[ti,:]
        w_ri_emb = self.W_R[ri,:]
        d_ri_emb = self.D_R[ri,:]
        
        #tj = random.sample(np.arange(0,self.num_docs).tolist(), hi_emb.shape[0])
        #if torch.cuda.is_available():
        #    tj = torch.LongTensor(np.asarray(tj)).to(torch.device('cuda'))
            
        tj_emb = self.d2v._D[tj,:]
        
        pos = None
        neg = None
        
        if self.kg_model_ver == 'transh':
            pos_h_e = projection_transH(hi_emb, d_ri_emb)
            pos_t_e = projection_transH(ti_emb, d_ri_emb)
            neg_h_e = projection_transH(hi_emb, d_ri_emb)
            neg_t_e = projection_transH(tj_emb, d_ri_emb)
        
            pos = torch.sum((pos_h_e + w_ri_emb - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + w_ri_emb - neg_t_e) ** 2, 1)
            
        elif self.kg_model_ver == 'transe':
            pos = torch.sum((hi_emb + w_ri_emb - ti_emb) ** 2, 1)
            neg = torch.sum((hi_emb + w_ri_emb - tj_emb) ** 2, 1)
        
        elif self.kg_model_ver == 'distmult':
            pos = torch.sum(projection_DistMult(w_ri_emb, hi_emb, ti_emb), 1) 
            neg = torch.sum(projection_DistMult(w_ri_emb, hi_emb, tj_emb), 1) 
        
        elif self.kg_model_ver == 'transr':
            M_R = self.M_R[ri,:]
            hi_emb = torch.einsum('ij, ijk -> ik', hi_emb, M_R)
            ti_emb = torch.einsum('ij, ijk -> ik', ti_emb, M_R)
            tj_emb = torch.einsum('ij, ijk -> ik', tj_emb, M_R)
            
            hi_emb = F.normalize(hi_emb, p=2, dim=1)
            ti_emb = F.normalize(ti_emb, p=2, dim=1)
            tj_emb = F.normalize(tj_emb, p=2, dim=1)
            
            pos = torch.sum((hi_emb + w_ri_emb - ti_emb) ** 2, 1)
            neg = torch.sum((hi_emb + w_ri_emb - tj_emb) ** 2, 1)
            
        elif self.kg_model_ver == 'transd':
            hi_emb = projection_transD(hi_emb, w_ri_emb)
            ti_emb = projection_transD(ti_emb, w_ri_emb)
            tj_emb = projection_transD(tj_emb, w_ri_emb)
            
            pos = torch.sum((hi_emb + w_ri_emb - ti_emb) ** 2, 1)
            neg = torch.sum((hi_emb + w_ri_emb - tj_emb) ** 2, 1)
        
        
        if self.d2v_model_ver != 'none':
            d2v_output = self.d2v.forward(context_ids, doc_ids, target_noise_ids)
            d2v_loss = self.cost_func.forward(d2v_output)
        else:
            d2v_output = torch.FloatTensor([0])
            d2v_loss = torch.FloatTensor([0])
            
        if self.kg_model_ver != 'none':
            #print (pos.shape, neg.shape)
            kg_loss = self.kg_loss_fn(pos, neg, self.margin)
        else:
            kg_loss = torch.FloatTensor([0])
            
        if self.d2v_model_ver != 'none' and self.kg_model_ver != 'none':
            total_loss = (1-self.delta)*d2v_loss + self.delta*kg_loss
        elif self.d2v_model_ver != 'none':
            total_loss = d2v_loss
        elif self.kg_model_ver != 'none':
            total_loss = kg_loss
        else:
            raise ValueError("Both D2V and KG model can not be none")
                
        return total_loss, d2v_loss, kg_loss, d2v_output, pos, neg
        
    def get_paragraph_vector(self, index):
        return self.d2v._D[index, :].data.tolist()
    
class DM(nn.Module):
    """Distributed Memory version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DM, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        # word matrix
        self._W = nn.Parameter(
            torch.randn(num_words, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        context_ids: torch.Tensor of size (batch_size, num_context_words)
            Vocabulary indices of context words.

        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # combine a paragraph vector with word vectors of
        # input (context) words
        x = torch.add(
            self._D[doc_ids, :], torch.sum(self._W[context_ids, :], dim=1))

        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            x.unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[index, :].data.tolist()


class DBOW(nn.Module):
    """Distributed Bag of Words version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DBOW, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            self._D[doc_ids, :].unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[index, :].data.tolist()
