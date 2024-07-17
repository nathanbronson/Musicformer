"""
    Musicformer: a neural network for unsupervised embeddings
    Copyright (C) 2023  Nathan Bronson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import json
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *
from transvae.opt import NoamOpt
from transvae.data import vae_data_gen, make_std_mask
from transvae.loss import vae_loss, trans_vae_loss


####### MODEL SHELL ##########

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """
    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        if 'BATCH_SIZE' not in self.params.keys():
            self.params['BATCH_SIZE'] = 500
        if 'BATCH_CHUNKS' not in self.params.keys():
            self.params['BATCH_CHUNKS'] = 5
        if 'BETA_INIT' not in self.params.keys():
            self.params['BETA_INIT'] = 1e-8
        if 'BETA' not in self.params.keys():
            self.params['BETA'] = 0.05
        if 'ANNEAL_START' not in self.params.keys():
            self.params['ANNEAL_START'] = 0
        if 'LR' not in self.params.keys():
            self.params['LR_SCALE'] = 1
        if 'WARMUP_STEPS' not in self.params.keys():
            self.params['WARMUP_STEPS'] = 10000
        if 'EPS_SCALE' not in self.params.keys():
            self.params['EPS_SCALE'] = 1
        if 'CHAR_DICT' in self.params.keys():
            self.vocab_size = len(self.params['CHAR_DICT'].keys())
            self.pad_idx = self.params['CHAR_DICT']['_']
            if 'CHAR_WEIGHTS' in self.params.keys():
                self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
            else:
                self.params['CHAR_WEIGHTS'] = torch.ones(self.vocab_size, dtype=torch.float)
        self.loss_func = vae_loss
        self.data_gen = vae_data_gen

        ### Sequence length hard-coded into model
        self.src_len = 126
        self.tgt_len = 125

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}
        self.loaded_from = None

    def save(self, state, fn, path='checkpoints', use_name=True):
        """
        Saves current model state to .ckpt file

        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                if self.name is not None:
                    fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        torch.save(state, save_path)

    def load(self, checkpoint_path):
        """
        Loads a saved model state

        Arguments:
            checkpoint_path (str, required): Path to saved .ckpt file
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.loaded_from = checkpoint_path
        for k in self.current_state.keys():
            try:
                self.current_state[k] = loaded_checkpoint[k]
            except KeyError:
                self.current_state[k] = None

        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        for k, v in self.current_state['params'].items():
            if k in self.arch_params or k not in self.params.keys():
                self.params[k] = v
            else:
                pass
        self.vocab_size = len(self.params['CHAR_DICT'].keys())
        self.pad_idx = self.params['CHAR_DICT']['_']
        self.build_model()
        self.model.load_state_dict(self.current_state['model_state_dict'])
        self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])

    def train(self, train_mols, val_mols, train_props=None, val_props=None,
              epochs=100, save=True, save_freq=None, log=True, log_dir='trials'):
        """
        Train model and validate

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
            train_props (np.array): Numpy array containing chemical property of
                                   molecular structure
            val_props (np.array): Same format as train_prop. Used for model
                                 development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
        """
        ### Prepare data iterators
        train_data = self.data_gen(train_mols, train_props, char_dict=self.params['CHAR_DICT'])
        val_data = self.data_gen(val_mols, val_props, char_dict=self.params['CHAR_DICT'])

        train_iter = torch.utils.data.DataLoader(train_data,
                                                 batch_size=self.params['BATCH_SIZE'],
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False, drop_last=True)
        val_iter = torch.utils.data.DataLoader(val_data,
                                               batch_size=self.params['BATCH_SIZE'],
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS']


        #torch.backends.cudnn.benchmark = True

        ### Determine save frequency
        if save_freq is None:
            save_freq = epochs

        ### Setup log file
        if log:
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_'+self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,kld_loss,prop_mse_loss,run_time\n')
            log_file.close()

        ### Initialize Annealer
        kl_annealer = KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                 epochs, self.params['ANNEAL_START'])

        ### Epoch loop
        for epoch in range(epochs):
            ### Train Loop
            self.model.train()
            losses = []
            beta = kl_annealer(epoch)
            for j, data in enumerate(train_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:-1]
                    props_data = batch_data[:,-1]
                    if self.use_gpu:
                        mols_data = mols_data.to("mps")
                        props_data = props_data.to("mps")


                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)

                    if self.model_type == 'transformer':
                        x_out, mu, logvar, pred_len, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_mse = trans_vae_loss(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.params['CHAR_WEIGHTS'],
                                                                            beta)
                        avg_bcemask_losses.append(bce_mask.item())
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        loss, bce, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                    loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                         j, 'train',
                                                                         avg_loss,
                                                                         avg_bce,
                                                                         avg_bcemask,
                                                                         avg_kld,
                                                                         avg_prop_mse,
                                                                         run_time))
                    log_file.close()
            train_loss = np.mean(losses)

            ### Val Loop
            self.model.eval()
            losses = []
            for j, data in enumerate(val_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:-1]
                    props_data = batch_data[:,-1]
                    if self.use_gpu:
                        mols_data = mols_data.to("mps")
                        props_data = props_data.to("mps")

                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)
                    scores = Variable(data[:,-1])

                    if self.model_type == 'transformer':
                        x_out, mu, logvar, pred_len, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_mse = trans_vae_loss(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.params['CHAR_WEIGHTS'],
                                                                            beta)
                        avg_bcemask_losses.append(bce_mask.item())
                    else:
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, src_mask, tgt_mask)
                        loss, bce, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  beta)
                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                losses.append(avg_loss)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'test',
                                                                avg_loss,
                                                                avg_bce,
                                                                avg_bcemask,
                                                                avg_kld,
                                                                avg_prop_mse,
                                                                run_time))
                    log_file.close()

            self.n_epochs += 1
            val_loss = np.mean(losses)
            print('Epoch - {} Train - {} Val - {} KLBeta - {}'.format(self.n_epochs, train_loss, val_loss, beta))

            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.current_state['best_loss'] = self.best_loss
                if save:
                    self.save(self.current_state, 'best')

            if (self.n_epochs) % save_freq == 0:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:
                    self.save(self.current_state, epoch_str)

    ### Sampling and Decoding Functions
    def sample_from_memory(self, size, mode='rand', sample_dims=None, k=5):
        """
        Quickly sample from latent dimension

        Arguments:
            size (int, req): Number of samples to generate in one batch
            mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
        Returns:
            z (torch.tensor): NxD_latent tensor containing sampled memory vectors
        """
        if mode == 'rand':
            z = torch.randn(size, self.params['d_latent'])
        else:
            assert sample_dims is not None, "ERROR: Must provide sample dimensions"
            if mode == 'top_dims':
                z = torch.zeros((size, self.params['d_latent']))
                for d in sample_dims:
                    z[:,d] = torch.randn(size)
            elif mode == 'k_dims':
                z = torch.zeros((size, self.params['d_latent']))
                d_select = np.random.choice(sample_dims, size=k, replace=False)
                for d in d_select:
                    z[:,d] = torch.randn(size)
        return z

    def greedy_decode(self, mem, src_mask=None, condition=[]):
        """
        Greedy decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens (if
                                     model_type == 'transformer')
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long()
        for tok in condition:
            condition_symbol = self.params['CHAR_DICT'][tok]
            condition_vec = torch.ones(mem.shape[0],1).fill_(condition_symbol).long()
            decoded = torch.cat([decoded, condition_vec], dim=1)
        tgt = torch.ones(mem.shape[0],max_len+1).fill_(start_symbol).long()
        tgt[:,:len(condition)+1] = decoded
        if src_mask is None and self.model_type == 'transformer':
            mask_lens = self.model.encoder.predict_mask_length(mem)
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len+1))
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i,:,:mask_len] = torch.ones((1, 1, mask_len))
        elif self.model_type != 'transformer':
            src_mask = torch.ones((mem.shape[0], 1, self.src_len))

        if self.use_gpu:
            src_mask = src_mask.to("mps")
            decoded = decoded.to("mps")
            tgt = tgt.to("mps")

        self.model.eval()
        for i in range(len(condition), max_len):
            if self.model_type == 'transformer':
                decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
                if self.use_gpu:
                    decode_mask = decode_mask.to("mps")
                out = self.model.decode(mem, src_mask, Variable(decoded),
                                        decode_mask)
            else:
                out, _ = self.model.decode(tgt, mem)
            out = self.model.generator(out)
            prob = F.softmax(out[:,i,:], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1
            tgt[:,i+1] = next_word
            if self.model_type == 'transformer':
                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
        decoded = tgt[:,1:]
        return decoded

    def reconstruct(self, data, method='greedy', log=True, return_mems=True, return_str=True):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            log (bool): If true, tracks reconstruction progress in separate log file
            return_mems (bool): If true, returns memory vectors in addition to decoded SMILES
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded_smiles (list): Decoded smiles data - either decoded SMILES strings or tensor of
                                   token ids
            mems (np.array): Array of model memory vectors
        """
        data = vae_data_gen(data, props=None, char_dict=self.params['CHAR_DICT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=True)
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']

        self.model.eval()
        decoded_smiles = []
        mems = torch.empty((data.shape[0], self.params['d_latent'])).cpu()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('calcs/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if self.use_gpu:
                    mols_data = mols_data.to("mps")
                    props_data = props_data.to("mps")

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    _, mem, _, _ = self.model.encode(src, src_mask)
                else:
                    _, mem, _ = self.model.encode(src)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()

                ### Decode logic
                if method == 'greedy':
                    decoded = self.greedy_decode(mem, src_mask=src_mask)
                else:
                    decoded = None

                if return_str:
                    decoded = decode_mols(decoded, self.params['ORG_DICT'])
                    decoded_smiles += decoded
                else:
                    decoded_smiles.append(decoded)

        if return_mems:
            return decoded_smiles, mems.detach().numpy()
        else:
            return decoded_smiles

    def sample(self, n, method='greedy', sample_mode='rand',
                        sample_dims=None, k=None, return_str=True,
                        condition=[]):
        """
        Method for sampling from memory and decoding back into SMILES strings

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            sample_mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded (list): Decoded smiles data - either decoded SMILES strings or tensor of
                            token ids
        """
        mem = self.sample_from_memory(n, mode=sample_mode, sample_dims=sample_dims, k=k)

        if self.use_gpu:
            mem = mem.to("mps")

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem, condition=condition)
        else:
            decoded = None

        if return_str:
            decoded = decode_mols(decoded, self.params['ORG_DICT'])
        return decoded

    def calc_mems(self, data, log=True, save_dir='memory', save_fn='model_name', save=True):
        """
        Method for calculating and saving the memory of each neural net

        Arguments:
            data (np.array, req): Input array containing SMILES strings
            log (bool): If true, tracks calculation progress in separate log file
            save_dir (str): Directory to store output memory array
            save_fn (str): File name to store output memory array
            save (bool): If true, saves memory to disk. If false, returns memory
        Returns:
            mems(np.array): Reparameterized memory array
            mus(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """
        data = vae_data_gen(data, props=None, char_dict=self.params['CHAR_DICT'])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=True)
        save_shape = len(data_iter)*self.params['BATCH_SIZE']
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']
        mems = torch.empty((save_shape, self.params['d_latent'])).cpu()
        mus = torch.empty((save_shape, self.params['d_latent'])).cpu()
        logvars = torch.empty((save_shape, self.params['d_latent'])).cpu()

        self.model.eval()
        for j, data in enumerate(data_iter):
            if log:
                log_file = open('memory/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data = batch_data[:,:-1]
                props_data = batch_data[:,-1]
                if self.use_gpu:
                    mols_data = mols_data.to("mps")
                    props_data = props_data.to("mps")

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)

                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    mem, mu, logvar, _ = self.model.encode(src, src_mask)
                else:
                    mem, mu, logvar = self.model.encode(src)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()
                mus[start:stop, :] = mu.detach().cpu()
                logvars[start:stop, :] = logvar.detach().cpu()

        if save:
            if save_fn == 'model_name':
                save_fn = self.name
            save_path = os.path.join(save_dir, save_fn)
            np.save('{}_mems.npy'.format(save_path), mems.detach().numpy())
            np.save('{}_mus.npy'.format(save_path), mus.detach().numpy())
            np.save('{}_logvars.npy'.format(save_path), logvars.detach().numpy())
        else:
            return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy()


####### Encoder, Decoder and Generator ############

class TransVAE(VAEShell):
    """
    Transformer-based VAE class. Between the encoder and decoder is a stochastic
    latent space. "Memory value" matrices are convolved to latent bottleneck and
    deconvolved before being sent to source attention in decoder.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128, d_ff=512,
                 d_latent=128, h=4, dropout=0.1, bypass_bottleneck=False,
                 property_predictor=False, d_pp=256, depth_pp=2, load_fn=None):
        super().__init__(params, name)
        """
        Instatiating a TransVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_ff (int): Dimensionality of feed-forward layers
            d_latent (int): Dimensionality of latent space
            h (int): Number of heads per attention layer
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
            property_predictor (bool): If true, model will predict property from latent memory
            d_pp (int): Dimensionality of property predictor layers
            depth_pp (int): Number of property predictor layers
            load_fn (str): Path to checkpoint file
        """

        ### Store architecture params
        self.model_type = 'transformer'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_ff'] = d_ff
        self.params['d_latent'] = d_latent
        self.params['h'] = h
        self.params['dropout'] = dropout
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_ff', 'd_latent', 'h', 'dropout', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            self.build_model()
        else:
            self.load(load_fn)
