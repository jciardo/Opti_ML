import dataclasses
import random 
from collections import defaultdict
import wandb
import time
import os
import einops
import numpy as np
import plotly.graph_objects as go



import torch as t
import torch.nn.functional as F
import torch.optim as optim

from model import Config, Transformer

#! Helpers

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    '''Helper function to plot multiple lines'''
    if type(lines_list)==t.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==t.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.upd

def calculate_coefficients(logits, fourier_basis, key_freqs, p, device):
    '''updated version from https://colab.research.google.com/drive/1ScVRL8OCtTFpOHpgfz0PLTFvX4g_YbuN?usp=sharing#scrollTo=WY4nPUDwl9UN
    '''
    x = t.arange(p)[None, :, None, None]
    y = t.arange(p)[None, None, :, None]
    z = t.arange(p)[None, None, None, :]
    w = t.arange(1, (p//2+1))[:, None, None, None]
    coses = t.cos(w*t.pi*2/p * (x + y - z)).to(device)
    coses = coses.reshape(p//2, p*p, p)
    coses/= coses.pow(2).sum([-2, -1], keepdim=True).sqrt()
    cos_coefficients = (coses * logits).sum([-2, -1])
    return cos_coefficients

def to_numpy(tensor, flat=False):
    if type(tensor)!=t.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def unflatten_first(tensor, p):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor

def extract_freq_2d(tensor, freq, p):
    # Takes in a pxpx... or batch x ... tensor, returns a 3x3x... tensor of the 
    # Linear and quadratic terms of frequency freq
    tensor = unflatten_first(tensor, p)
    # Extracts the linear and quadratic terms corresponding to frequency freq
    index_1d = [0, 2*freq-1, 2*freq]
    # Some dumb manipulation to use fancy array indexing rules
    # Gets the rows and columns in index_1d
    return tensor[[[i]*3 for i in index_1d], [index_1d]*3]

def fft2d(mat, p, fourier_basis):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = t.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)

def make_fourier_basis(config: Config):
    fourier_basis = []
    fourier_basis.append(t.ones(config.p)/np.sqrt(config.p))
    fourier_basis_names = ['Const']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie 
    # alternating +1 and -1
    for i in range(1, config.p//2 +1):
        fourier_basis.append(t.cos(2*t.pi*t.arange(config.p)*i/config.p))
        fourier_basis.append(t.sin(2*t.pi*t.arange(config.p)*i/config.p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    return t.stack(fourier_basis, dim=0).to(config.device)

def calculate_key_freqs(config: Config, model: Transformer, all_data):
    # TODO this was moved from the app code; probably move it around
    labels = t.tensor([config.fn(i, j) for i, j, _ in all_data]).to(config.device)
    cache = {}
    model.remove_all_hooks() # TODO is this line fucky??
    model.cache_all(cache)
    model(all_data)
    neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1]
    # Center the neurons to remove the constant term
    neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    # Note that fourier_neuron_acts[(0, 0), i]==0 for all i, because we centered the activations
    fourier_basis = make_fourier_basis(config = config)
    fourier_neuron_acts = fft2d(neuron_acts_centered, p = config.p, fourier_basis=fourier_basis)

    fourier_neuron_acts_square = fourier_neuron_acts.reshape(config.p, config.p, config.d_mlp)
    neuron_freqs = []
    neuron_frac_explained = []
    for ni in range(config.d_mlp):
        best_frac_explained = -1e6
        best_freq = -1
        for freq in range(1, config.p//2):
            # We extract the linear and quadratic fourier terms of frequency freq, 
            # and look at how much of the variance of the full vector this explains
            # If neurons specialise into specific frequencies, one frequency should 
            # have a large value
            numerator = extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq, p = config.p).pow(2).sum()
            denominator = fourier_neuron_acts_square[:, :, ni].pow(2).sum().item()
            frac_explained = numerator / denominator
            if frac_explained > best_frac_explained:
                best_freq = freq
                best_frac_explained = frac_explained
        neuron_freqs.append(best_freq)
        neuron_frac_explained.append(best_frac_explained)
    neuron_freqs = np.array(neuron_freqs)
    neuron_frac_explained = to_numpy(neuron_frac_explained)
    key_freqs, neuron_freq_counts = np.unique(neuron_freqs, return_counts=True)
    return key_freqs

def test_logits(logits, p, is_train, is_test, labels, bias_correction=False, original_logits=None, mode='all'):
    # Calculates cross entropy loss of logits representing a batch of all p^2 
    # possible inputs
    # Batch dimension is assumed to be first
    if logits.shape[1]==p*p:
        logits = logits.T
    if logits.shape==t.Size([p*p, p+1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p*p, p)
    if bias_correction:
        # Applies bias correction - we correct for any missing bias terms, 
        # independent of the input, by centering the new logits along the batch 
        # dimension, and then adding the average original logits across all inputs
        logits = einops.reduce(original_logits - logits, 'batch ... -> ...', 'mean') + logits
    if mode=='train':
        return cross_entropy_high_precision(logits[is_train], labels[is_train])
    elif mode=='test':
        return cross_entropy_high_precision(logits[is_test], labels[is_test])
    elif mode=='all':
        return cross_entropy_high_precision(logits, labels)

def get_2d_fourier_component(tensor, x, y, fourier_basis):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component 
    # (x, y)
    vec = fourier_2d_basis_term(x, y, fourier_basis).flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)

def fourier_2d_basis_term(x_index, y_index, fourier_basis):
    # Returns the 2D Fourier basis term corresponding to the outer product of 
    # the x_index th component in the x direction and y_index th component in the 
    # y direction
    # Returns a 1D vector of length p^2
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def get_component_cos_xpy(tensor, freq, fourier_basis, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1, fourier_basis=fourier_basis).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq, fourier_basis=fourier_basis).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)
    
def get_component_sin_xpy(tensor, freq, fourier_basis, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1, fourier_basis=fourier_basis).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq, fourier_basis=fourier_basis).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)

def calculate_excluded_loss(config: Config, fourier_basis, key_freqs, is_train, is_test, labels, logits):
    row = []
    for freq in key_freqs:
        cos = get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis) 
        sin = get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis) 
        value = test_logits(logits - cos - sin, bias_correction=False, mode='train', p = config.p,
           is_train = is_train, is_test = is_test, labels = labels)
        row.append(value.item())
    return row

def get_components_of_trig_loss(logits, freq, fourier_basis):
    cos = get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis)
    sin = get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis)
    return cos + sin

def calculate_trig_loss(config: Config, model, train, logits, key_freqs, fourier_basis, all_data, is_train, is_test, labels, mode='all'):
    trig_logits = sum([get_components_of_trig_loss(logits, freq, fourier_basis) for freq in key_freqs])
    return test_logits(trig_logits, 
                        p = config.p,
                        is_train = is_train, 
                        is_test = is_test,
                        labels = labels,
                        bias_correction=True, 
                        original_logits=logits, 
                        mode=mode)

def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(t.float32), dim=-1)
    prediction_logprobs = t.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -t.mean(prediction_logprobs)
    return loss

def gen_train_test(config: Config):
    '''Generate train and test split'''
    num_to_generate = config.p
    pairs = [(i, j, num_to_generate) for i in range(num_to_generate) for j in range(num_to_generate)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train*len(pairs))
    return pairs[:div], pairs[div:]

def full_loss(config : Config, model: Transformer, data):
    '''Takes the cross entropy loss of the model on the data'''
    # Take the final position only
    logits = model(data)[:, -1]
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    return cross_entropy_high_precision(logits, labels)

class Trainer:
    '''TODO
    ways this stinks:
    - callbacks every k epochs 
    - training on infinite data
    - general abstract class w/o assumption and subclasses w/ more assumptions
    - check out hugging face trainer
    - disentangle optimization step and taking gradients
    - forward compatibility, e.g. batches per step
    '''

    def __init__(self, config : Config, model = None) -> None:
        wandb.init(project = "grokking", config = dataclasses.asdict(config))
        self.model = model if model is not None else Transformer(config, use_cache=False)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1)) # TODO make this a config option
        self.run_name = f"grok_{int(time.time())}"
        self.train, self.test = gen_train_test(config = config)
        self.metrics_dictionary = defaultdict(dict) # so we can safely call 'update' on keys
        print('training length = ', len(self.train))
        print('testing length = ', len(self.test))
        self.train_losses = []
        self.test_losses = []
        self.config = config

    def save_epoch(self, epoch, save_to_wandb = True):
        ''' precondition! train loss and test losses have been appended to '''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)
            print("Saved epoch to wandb")
        #if self.config.save_models: 
            #t.save(save_dict, helpers.root/self.run_name/f"{epoch}.pth")
            #print(f"Saved model to {helpers.root/self.run_name/f'{epoch}.pth'}")
        #self.metrics_dictionary[epoch].update(save_dict)

    def do_a_training_step(self, epoch: int):
        '''returns train_loss, test_loss'''
        train_loss = full_loss(config = self.config, model = self.model, data = self.train)
        test_loss = full_loss(config = self.config, model = self.model, data = self.test)
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        if epoch % 100 == 0:
            # TODO is this ok? this was np.log, and it was barking at me ; i think np.log was being interpreted as a logging module
            print(f'Epoch {epoch}, train loss {t.log(train_loss).item():.4f}, test loss {t.log(test_loss).item():.4f}')
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return train_loss, test_loss

    def initial_save_if_appropriate(self):
        '''if self.config.save_models:
            os.mkdir(helpers.root/self.run_name)
            save_dict = {
                'model': self.model.state_dict(),
                'train_data' : self.train,
                'test_data' : self.test}
            t.save(save_dict, helpers.root/self.run_name/'init.pth')'''
    



    def post_training_save(self, save_optimizer_and_scheduler = True, log_to_wandb = True):
        '''if not self.config.save_models:
            os.makedirs(helpers.root/self.run_name, exist_ok=True)
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'epoch': self.config.num_epochs,
        }
        if save_optimizer_and_scheduler:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        if log_to_wandb:
            wandb.log(save_dict)
        t.save(save_dict, helpers.root/self.run_name/f"final.pth")
        print(f"Saved model to {helpers.root/self.run_name/f'final.pth'}")
        self.metrics_dictionary[save_dict['epoch']].update(save_dict)'''




    def take_metrics(self, train, epoch):
        with t.inference_mode():
            def sum_sq_weights():
                # TODO refactor- taken from app code
                row = []
                for name, param in self.model.named_parameters():
                    row.append(param.pow(2).sum().item())
                return row

            print('taking metrics')

            all_data = t.tensor([(i, j, self.config.p) for i in range(self.config.p) for j in range(self.config.p)]).to(self.config.device)
            # TODO calculate key freqs is the most expensive part of this
            key_freqs = calculate_key_freqs(config = self.config, model = self.model, all_data = all_data) 
            logits = self.model(all_data)[:, -1, :-1] # TODO i think this is equivalent to what's in the new paper?
            fourier_basis = make_fourier_basis(config = self.config)
            is_train, is_test = self.config.is_train_is_test(train = train)
            labels = t.tensor([self.config.fn(i, j) for i, j, _ in all_data]).to(self.config.device)

            metrics = {
                'epoch': epoch, 
                'trig_loss': calculate_trig_loss(config = self.config,
                    model = self.model,
                    train = train,
                    key_freqs = key_freqs,
                    is_test=is_test,
                    is_train=is_train,
                    labels=labels,
                    logits = logits,
                    fourier_basis=fourier_basis, 
                    all_data=all_data),
                'sum_of_squared_weights': sum_sq_weights(),
                'excluded_loss': calculate_excluded_loss(
                    logits = logits,
                    key_freqs = key_freqs,
                    fourier_basis = fourier_basis,
                    is_train=is_train,
                    config = self.config,
                    is_test = is_test,
                    labels=labels),
                'coefficients': calculate_coefficients(p = self.config.p, logits = logits, fourier_basis = fourier_basis, key_freqs = key_freqs, device = self.config.device),
            }
            wandb.log(metrics)
            print("Logged metrics to wandb")
            self.metrics_dictionary[epoch].update(metrics)

def train_model(config: Config):
    world = Trainer(config = config)
    print(f'Run name {world.run_name}')
    world.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if test_loss.item() < config.stopping_thresh:
            break
        if config.is_it_time_to_save(epoch = epoch):
            # TODO this also used to do a check about test loss- pretty sure not necessary
            world.save_epoch(epoch = epoch)
        if config.is_it_time_to_take_metrics(epoch = epoch):
            world.take_metrics(epoch = epoch, train = world.train)

    world.post_training_save(save_optimizer_and_scheduler=True)
    lines([world.train_losses, world.test_losses], labels=['train', 'test'], log_y=True)
    return world # to export the dictionary with the training metrics