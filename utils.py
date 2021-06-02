import os
from os import remove
from os.path import join, dirname, isfile

import matplotlib.pyplot as plt
import torch
import numpy as np

_root_dir = dirname(__file__)

DATA_DIR = join(_root_dir, 'INPUT')
MODELS_DIR = join(_root_dir, 'OUTPUT')
_DIAGNOSTICS_DIR = join(_root_dir, 'OUTPUT')

try:
    os.makedirs(MODELS_DIR)
except:
    pass
    
try:
    os.makedirs(_DIAGNOSTICS_DIR)
except:
    pass
    
_MODEL_NAME = ("{:s}_D2V.{:s}_KG.{:s}_contextsize.{:d}_numnoisewords.{:d}"
                  "_vecdim.{:d}_batchsize.{:d}_lr.{:f}_epoch.{:d}_loss.{:f}"
                  ".pth")
_DOC_EMB_FILE = ("doc_emb.npy")
_REL_EMB_FILE = ("rel_emb.npy")
_REL_NORM_EMB_FILE = ("rel_norm_emb.npy")
                                    
_DIAGNOSTIC_FILE_NAME = ("{:s}_D2V.{:s}_KG.{:s}_contextsize.{:d}_numnoisewords.{:d}"
                  "_vecdim.{:d}_batchsize.{:d}_lr.{:f}"
                  ".csv")


def save_training_state(data_file_name,
                        d2v_model_ver,
                        kg_model_ver,
                        vec_combine_method,
                        context_size,
                        num_noise_words,
                        vec_dim,
                        batch_size,
                        lr,
                        epoch_i,
                        loss,
                        model_state,
                        model,
                        save_all,
                        generate_plot,
                        is_best_loss,
                        prev_model_file_path,
                        d2v_model_ver_is_dbow):
    """Saves the state of the model. If generate_plot is True, it also
    saves current epoch's loss value and generates a plot of all loss
    values up to this epoch.

    Returns
    -------
        str representing a model file path from the previous epoch
    """
    if generate_plot:
        # save the loss value for a diagnostic plot
        diagnostic_file_name = _DIAGNOSTIC_FILE_NAME.format(
                data_file_name[:-4],
                d2v_model_ver,
                kg_model_ver,
                context_size,
                num_noise_words,
                vec_dim,
                batch_size,
                lr)

        diagnostic_file_path = join(_DIAGNOSTICS_DIR, diagnostic_file_name)

        if epoch_i == 0 and isfile(diagnostic_file_path):
            remove(diagnostic_file_path)

        with open(diagnostic_file_path, 'a') as f:
            f.write('{:f}\n'.format(loss))

        # generate a diagnostic loss plot
        with open(diagnostic_file_path) as f:
            loss_values = [float(l.rstrip()) for l in f.readlines()]

        diagnostic_plot_file_path = diagnostic_file_path[:-3] + 'png'
        fig = plt.figure()
        plt.plot(range(1, epoch_i + 2), loss_values, color='r')
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        fig.savefig(diagnostic_plot_file_path, bbox_inches='tight')
        plt.close()

    # save the model
    model_file_name = _MODEL_NAME.format(
        data_file_name[:-4],
        d2v_model_ver,
        kg_model_ver,
        context_size,
        num_noise_words,
        vec_dim,
        batch_size,
        lr,
        epoch_i + 1,
        loss)

    model_file_path = join(MODELS_DIR, model_file_name)

    if save_all:
        torch.save(model_state, model_file_path)
        np.save(join(MODELS_DIR, _DOC_EMB_FILE), model.d2v._D.detach().cpu().numpy())
        if kg_model_ver == 'transr':
            np.save(join(MODELS_DIR, _REL_EMB_FILE), model.M_R.detach().cpu().numpy())
        else:
            np.save(join(MODELS_DIR, _REL_EMB_FILE), model.W_R.detach().cpu().numpy())
        if kg_model_ver == 'transh':
            np.save(join(MODELS_DIR, _REL_NORM_EMB_FILE), model.D_R.detach().cpu().numpy())
            
        return None
    elif is_best_loss:
        if prev_model_file_path is not None:
            remove(prev_model_file_path)
        torch.save(model_state, model_file_path)
        np.save(join(MODELS_DIR, _DOC_EMB_FILE), model.d2v._D.detach().cpu().numpy())
        if kg_model_ver == 'transr':
            np.save(join(MODELS_DIR, _REL_EMB_FILE), model.M_R.detach().cpu().numpy())
        else:
            np.save(join(MODELS_DIR, _REL_EMB_FILE), model.W_R.detach().cpu().numpy())
        if kg_model_ver == 'transh':
            np.save(join(MODELS_DIR, _REL_NORM_EMB_FILE), model.D_R.detach().cpu().numpy())
            
        return model_file_path
    else:
        return prev_model_file_path
