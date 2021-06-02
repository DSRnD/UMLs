import time
from sys import float_info, stdout
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import pickle

import pandas as pd
import fire
import torch
from torch.optim import Adam
import numpy as np
import random

from data import load_dataset, NCEData, _NCEGenerator, _NCEGeneratorState, load_rel_dataset
from loss import NegativeSampling, marginLoss
from models import DM, DBOW, D2V_KG
from utils import save_training_state

def flatten(elems):
    return [e for elem in elems for e in elem]
    
def generate_negative_samples(all_relations,node_tensor,rel_tensor, n_nodes):
    batch_size = node_tensor.shape[0]
    output = []
    nodes = node_tensor.cpu().numpy().tolist()
    relations = rel_tensor.cpu().numpy().tolist()
    for i, node in enumerate(nodes):
        df_ = all_relations[all_relations['h'] == node]
        all_tails = set(df_[df_['r'] == relations[i]])
        #print (node, relations[i])
        #val = random.sample(list(set(np.arange(0,n_nodes).tolist()) - set(all_relations[node][relations[i]])), 1)
        val = random.sample(list(set(np.arange(0,n_nodes).tolist()) - all_tails), 1)
        output.append(val)
        
    return torch.LongTensor(np.asarray(output)[:,0])
    
def start(data_file_name,
          rel_data_file_name,
          val_rel_data_file_name,
          all_relations_file_name,
          num_noise_words,
          vec_dim,
          num_epochs,
          batch_size,
          lr,
          early_stopping_rounds=5,
          margin=0.5,
          delta=.25,
          d2v_model_ver='dbow',
          kg_model_ver='transh',
          context_size=2,
          vec_combine_method='sum',
          save_all=False,
          generate_plot=True,
          max_generated_batches=5,
          num_workers=1):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    d2v_model_ver: str, one of ('dm', 'dbow'), default='dbow'
        Version of the model as proposed by Q. V. Le et al., Distributed
        Representations of Sentences and Documents. 'dbow' stands for
        Distributed Bag Of Words, 'dm' stands for Distributed Memory.

    kg_model_ver: str, one of ('transe', 'transh', 'transr', 'transd', distmult', 'none'), default='transh'
        If no Relational model is used, 'none' can be selected
        
    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors when d2v_model_ver='dm'.
        Currently only the 'sum' operation is implemented.

    context_size: int, default=2
        Half the size of a neighbourhood of target words when d2v_model_ver='dm'
        (i.e. how many words left and right are regarded as context). When
        d2v_model_ver='dm' context_size has to greater than 0, when
        d2v_model_ver='dbow' context_size has to be 0.

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    lr: float
        Learning rate of the Adam optimizer.

    save_all: bool, default=False
        Indicates whether a checkpoint is saved after each epoch.
        If false, only the best performing model is saved.

    generate_plot: bool, default=True
        Indicates whether a diagnostic plot displaying loss value over
        epochs is generated after each epoch.

    max_generated_batches: int, default=5
        Maximum number of pre-generated batches.

    num_workers: int, default=1
        Number of batch generator jobs to run in parallel. If value is set
        to -1 number of machine cores are used.
    """
    if d2v_model_ver not in ('dm', 'dbow', 'none'):
        raise ValueError("Invalid version of the model")
        
    if kg_model_ver not in ('transe', 'transh', 'transr', 'transd', 'distmult', 'none'):
        raise ValueError("Invalid version of the model")

    d2v_model_ver_is_dbow = d2v_model_ver == 'dbow'

    if d2v_model_ver_is_dbow and context_size != 0:
        raise ValueError("Context size has to be zero when using dbow")
    if not d2v_model_ver_is_dbow:
        if vec_combine_method not in ('sum', 'concat'):
            raise ValueError("Invalid method for combining paragraph and word "
                             "vectors when using dm")
        if context_size <= 0:
            raise ValueError("Context size must be positive when using dm")

    dataset = load_dataset(data_file_name)
    rel_dict, n_rel = load_rel_dataset(rel_data_file_name, len(dataset))
    val_rel_dict, _ = load_rel_dataset(val_rel_data_file_name, len(dataset))
    
    #rel_df_val = pd.read_csv(val_rel_data_file_name, sep='\t', header=None)
    #val_dataset = torch.utils.data.TensorDataset(torch.LongTensor(rel_df_val.values[:,1:4].astype(int)))
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    
    all_relations = pd.read_csv(all_relations_file_name, header=None, sep='\t') #pickle.load(open(all_relations_file_name, 'rb'))
    all_relations.columns = ['desc','h','t','r']
    #nce_data = NCEData(
    #    dataset,
    #    batch_size,
    #    context_size,
    #    num_noise_words,
    #    max_generated_batches,
    #    num_workers)
    #nce_data.start()
    
    nce_data = _NCEGenerator(
            dataset,
            batch_size,
            context_size,
            num_noise_words,
            _NCEGeneratorState(context_size))
    
    #if torch.cuda.is_available():
    #    nce_data = nce_data.cuda_()
        
    #try:
    _run(data_file_name, dataset, all_relations, rel_dict, val_rel_dict, n_rel, nce_data, len(nce_data),
         nce_data.vocabulary_size(), context_size, num_noise_words, vec_dim,
         num_epochs, batch_size, lr, early_stopping_rounds, margin, delta, d2v_model_ver, kg_model_ver, vec_combine_method,
         save_all, generate_plot, d2v_model_ver_is_dbow)
    #except KeyboardInterrupt:
    #    nce_data.stop()
    
def _run(data_file_name,
         dataset,
         all_relations,
         rel_dict,
         val_rel_dict,
         n_rel,
         data_generator,
         num_batches,
         vocabulary_size,
         context_size,
         num_noise_words,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         early_stopping_rounds,
         margin,
         delta,
         d2v_model_ver,
         kg_model_ver,
         vec_combine_method,
         save_all,
         generate_plot,
         d2v_model_ver_is_dbow):

    #if d2v_model_ver_is_dbow:
    #    model = DBOW(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)
    #else:
    #    model = DM(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)
    #print (rel_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_relations = sum([len(values) for key,values in rel_dict.items()])
    
    model = D2V_KG(vec_dim, len(dataset), vocabulary_size, n_rel, d2v_model_ver, kg_model_ver, margin, delta)
    
    #cost_func = NegativeSampling().to(device)
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        model.cuda()

    print("Dataset comprised of {:d} documents and total {:d} relationships".format(len(dataset), total_relations))
    print("Vocabulary size is {:d}.\n".format(vocabulary_size))
    print("Training started.")

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    prev_model_file_path = None
    bad_epochs = 0
    #kg_loss_fn = marginLoss().to(device)
    
    for epoch_i in range(num_epochs):
        if bad_epochs < early_stopping_rounds:
            epoch_start_time = time.time()
            loss = []
            d2v_losses = []
            kg_losses = []
            
            val_loss = []
            val_d2v_losses = []
            val_kg_losses = []
            
            for batch_i in range(num_batches):
                batch = data_generator.next() #next(data_generator)
                rel_batch = torch.LongTensor(flatten([rel_dict[i] for i in torch.unique_consecutive(batch.doc_ids).cpu().numpy()]))
                
                if torch.cuda.is_available():
                    batch.cuda_()
                    rel_batch = rel_batch.cuda()
                #print (batch.doc_ids)
                
                tj = torch.LongTensor(np.asarray(random.sample(np.arange(0,len(dataset)).tolist(), rel_batch.shape[0])))
                #tj = generate_negative_samples(all_relations,rel_batch[:,0],rel_batch[:,2], len(dataset))
                
                #print (rel_batch.shape, tj.shape)
                
                total_loss, d2v_loss, kg_loss, x, pos, neg = model.forward(batch.context_ids,batch.doc_ids,batch.target_noise_ids,rel_batch[:,0].to(device), \
                                                       rel_batch[:,1].to(device), rel_batch[:,2].to(device), tj.to(device))
                
                #if d2v_model_ver_is_dbow:
                #    x = model.forward(batch.doc_ids, batch.target_noise_ids)
                #else:
                #    x = model.forward(
                #        batch.context_ids,
                #        batch.doc_ids,
                #        batch.target_noise_ids)
                
                '''
                if d2v_model_ver != 'none':
                    d2v_loss = cost_func.forward(x)
                else:
                    d2v_loss = torch.FloatTensor([0])
                
                if kg_model_ver != 'none':
                    kg_loss = kg_loss_fn(pos, neg, torch.FloatTensor(np.asarray([margin]*pos.shape[0])).to(device))
                else:
                    kg_loss = torch.FloatTensor([0])
                    
                if d2v_model_ver != 'none' and kg_model_ver != 'none':
                    total_loss = (1-delta)*d2v_loss + delta*kg_loss
                elif d2v_model_ver != 'none':
                    total_loss = d2v_loss
                elif kg_model_ver != 'none':
                    total_loss = kg_loss
                else:
                    raise ValueError("Both D2V and KG model can not be none")
                '''
                
                loss.append(total_loss.item())
                d2v_losses.append(d2v_loss.item())
                kg_losses.append(kg_loss.item())
                model.zero_grad()
                total_loss.backward()
                optimizer.step()

                rel_batch = torch.LongTensor(flatten([val_rel_dict[i] for i in torch.unique_consecutive(batch.doc_ids).cpu().numpy()]))
                
                try:
                    if torch.cuda.is_available():
                        batch.cuda_()
                        rel_batch = rel_batch.cuda()
                    #print (batch.doc_ids)
                    
                    tj = torch.LongTensor(np.asarray(random.sample(np.arange(0,len(dataset)).tolist(), rel_batch.shape[0])))
                    #tj = generate_negative_samples(all_relations,rel_batch[:,0],rel_batch[:,2], len(dataset))
                    
                    #print (rel_batch.shape, tj.shape)
                    
                    with torch.no_grad():
                        total_loss, d2v_loss, kg_loss, x, pos, neg = model.forward(batch.context_ids,batch.doc_ids,batch.target_noise_ids,rel_batch[:,0].to(device), \
                                                           rel_batch[:,1].to(device), rel_batch[:,2].to(device), tj.to(device))
                    
                    val_loss.append(total_loss.item())
                    val_d2v_losses.append(d2v_loss.item())
                    val_kg_losses.append(kg_loss.item())

                except:
                    pass
                    
                _print_progress(epoch_i, batch_i, num_batches)
                
            # end of epoch
            train_loss = torch.mean(torch.FloatTensor(loss))
            train_kg_loss = torch.mean(torch.FloatTensor(kg_losses))
            train_d2v_loss = torch.mean(torch.FloatTensor(d2v_losses))
            
            try:
                val_loss = torch.mean(torch.FloatTensor(val_loss))
                val_kg_loss = torch.mean(torch.FloatTensor(val_kg_losses))
                val_d2v_loss = torch.mean(torch.FloatTensor(val_d2v_losses))
            except:
                val_loss, val_kg_loss, val_d2v_loss = float("inf"), float("inf"), float("inf")
                
            if val_loss < best_val_loss:
                bad_epochs = 0
            else:
                bad_epochs += 1
                
            is_best_loss = val_loss < best_val_loss
            best_train_loss = min(train_loss, best_train_loss)
            best_val_loss = min(val_loss, best_val_loss)
            
            state = {
                'epoch': epoch_i + 1,
                'model_state_dict': model.state_dict(),
                'best_train_loss': best_train_loss,
                'best_val_loss': best_val_loss,
                'optimizer_state_dict': optimizer.state_dict()
            }

            prev_model_file_path = save_training_state(
                data_file_name,
                d2v_model_ver,
                kg_model_ver,
                vec_combine_method,
                context_size,
                num_noise_words,
                vec_dim,
                batch_size,
                lr,
                epoch_i,
                val_loss,
                state,
                model,
                save_all,
                generate_plot,
                is_best_loss,
                prev_model_file_path,
                d2v_model_ver_is_dbow)

            epoch_total_time = round(time.time() - epoch_start_time)
            
            print("({:d}s) - train loss: {:.4f}, train d2v_loss: {:.4f}, train kg_loss: {:.4f}  val loss: {:.4f}, val d2v_loss: {:.4f}, val kg_loss: {:.4f}".format(\
                epoch_total_time, train_loss, train_d2v_loss, train_kg_loss, val_loss, val_d2v_loss, val_kg_loss))

        else:
            pass

def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end='')
    stdout.write(" - {:d}%".format(progress))
    stdout.flush()


if __name__ == '__main__':
    fire.Fire(start)
