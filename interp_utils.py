import os
import sys
import random
import copy
import itertools
import functools
from typing import List, Optional, Tuple, Union

# Third-party Library Imports
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT

import einops
from fancy_einsum import einsum

from jaxtyping import Float, Int

from pathlib import Path

from IPython.display import display

import plotly.express as px

import datasets

from pprint import pprint as pp

from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import to_numpy

import sys
# sys.path.append(".")
sys.path.append("/datadrive/madhur/world-models/src")
# from training_utils import get_mlp_layers
import eval_utils

# Custom and Specific Library Imports
# import transformer_lens
# from transformer_lens import (
#     utils,
#     HookedTransformer,
#     HookedTransformerConfig,
#     FactoredMatrix,
#     ActivationCache,
# )
# from transformer_lens.hook_points import HookPoint
# from transformer_lens.utils import to_numpy

from transformers import (
    pipeline, 
    set_seed,
    AutoModelForCausalLM,
    GPT2Config,
    PreTrainedTokenizerFast,
    AutoConfig,
    GPT2LMHeadModel
)
from accelerate import Accelerator, DistributedType
import tokenizers

import circuitsvis as cv

# from transformers import LlamaForCausalLM, LlamaTokenizer

def get_tokenizer_from_ckpt_path(ckpt_path):
    # wm_tokenizer = tokenizers.Tokenizer.from_file("wm_tokenizer_temp.json")
    # tokenizer = PreTrainedTokenizerFast(tokenizer_object=wm_tokenizer)
    # tokenizer.model_max_length = config.n_positions

    tokenizer = PreTrainedTokenizerFast.from_pretrained(ckpt_path)
    return tokenizer

def acclerator_load_model(accelerator, model, checkpoint_path):
    model = accelerator.prepare(model) # prepare other objects (such as optimizers, LR schedulers etc.) if you want to load them as well
    accelerator.load_state(checkpoint_path)
    model = accelerator.unwrap_model(model)
    return model

def load_hooked_gpt2(ckpt_path, device, from_pretrained_kwargs, fold_ln=False):
    # MODEL_PATH: str = '/home/t-sgolechha/Desktop/auto-prompt/hf_checkpoints_7b_chat'
    # tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    # src/tokenizer_local/wm_tokenizer_simpl_no_trans_only_person_ques
    # hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
    # ckpt_path = "/datadrive/madhur/world-models/src/training_outputs/FromSingularity/1cd24e9d-9a34-4da0-987e-527619b49ec8"
    accelerator = Accelerator()
    hf_model, config = eval_utils.get_model_from_checkpoint_path(ckpt_path, 25, accelerator, args=None, step_or_epoch="epoch")

    tokenizer = get_tokenizer_from_ckpt_path(ckpt_path)
    assert tokenizer.model_max_length == config.n_positions
    assert config.pad_token_id == tokenizer.pad_token_id
    # breakpoint()
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model, device="cpu", \
                                              fold_ln=fold_ln, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer, **from_pretrained_kwargs)
    model = model.to(device)
    print('Successfully loaded hooked gpt2 model.')
    return model, tokenizer

# def get_mlp_layers(lyr_str: str, n_layers: int):
#     mlp_layers_list = None
#     if lyr_str == "all":
#         mlp_layers_list = list(range(n_layers))
#     elif lyr_str == "none":
#         mlp_layers_list = []
#     else:
#         mlp_layers_list = [int(i) for i in lyr_str.split()]
#         for i in mlp_layers_list:
#             assert 0 <= i < n_layers, f"Some specified layer for the parameter 'mlp_layers' is out of bounds [0, {n_layers-1}]."

#     return mlp_layers_list

# def get_model_from_checkpoint_path(ckpt_path, step_num, accelerator, args=None, step_or_epoch="step", mlp_layers=None, is_no_pos_encode=False):

#     if step_or_epoch == "step":
#         ckpt_dir = os.path.join(ckpt_path, "step_" + str(step_num))
#     elif step_or_epoch == "epoch":
#         ckpt_dir = os.path.join(ckpt_path, "epoch_" + str(step_num))

#     try:
#         config = AutoConfig.from_pretrained(os.path.join(ckpt_path, "config.json"))
#     except:
#         config = AutoConfig.from_pretrained(
#             "gpt2",
#             # n_positions=2 * n_positions,
#             n_embd=args.model_n_embd,
#             n_layer=args.model_n_layer,
#             n_head=args.model_n_head,
#             # resid_pdrop=0.0,
#             # embd_pdrop=0.0,
#             # attn_pdrop=0.0,
#             # use_cache=False,
#             vocab_size=args.vocab_size,
#             bos_token_id=args.bos_token_id, 
#             eos_token_id=args.eos_token_id,
#             pad_token_id=args.pad_token_id,
#             # sep_token_id=tokenizer.sep_token_id,
#             n_ctx=args.block_size,
#             # n_positions=1024
#         )

#     if mlp_layers is None:
#         mlp_layers_list = list(range(config['n_layer']))
#     else:
#         mlp_layers_list = get_mlp_layers(mlp_layers, config['n_layer'])
    
#     custom_args = {
#         "is_pos_encode": not is_no_pos_encode, 
#         "mlp_layers": mlp_layers_list
#     }
#     # config = GPT2Config(
#     #             n_embd=256,
#     #             n_layer=12,
#     #             n_head=8,
#     #         )

#     # model_untrained = AutoModelForCausalLM.from_config(config)
#     model_untrained = GPT2LMHeadModelCustom(config, custom_args)
#     model = acclerator_load_model(accelerator, model_untrained, ckpt_dir)
#     print(f"Loaded model's vocab size: {config.vocab_size}")
#     return model, config

def plot_heatmap(
        tensor, 
        labels, 
        title, 
        sizex=20, 
        sizey=13, 
        show_values=True, 
        show_labels=True,
        save=None,
        auto_limits=False,
        ):
    array = tensor.cpu().numpy()
    layers, heads = int(array.shape[0]), int(array.shape[1])
    ax = plt.figure(figsize=(sizex, sizey))
    # blue to white
    cmap = plt.cm.plasma
    # cmap = plt.cm.RdBu
    # tell imshow about color map so that only set colors are used
    if auto_limits:
        limits = (np.min(array), np.max(array))
        im = plt.imshow(array, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='nearest')
    else:
        # 0 to 1
        limits = (0, 1)
        im = plt.imshow(array, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='nearest')
    # use a blue-white-red colorbar
    im.set_clim(limits[0], limits[1])
    # show values in each cell
    if show_values:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                text = plt.text(j, i, round(array[i, j].item(), 2), ha="center", va="center", color="black")
    # perpendicular xlabels and ylabels from labels
    if show_labels:
        plt.xticks(np.arange(0, heads, 1), labels=labels[0])
        plt.yticks(np.arange(0, layers, 1), labels=labels[1])
        # perpendicular ytick labels
        plt.xticks(rotation=90)
    im_ratio = array.shape[0]/array.shape[1] 
    plt.colorbar(im,fraction=0.048*im_ratio, pad=0.14)
    plt.tick_params(axis='both', which='both', length=0)
    plt.title(title)
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

def plot_line(
        tensor,
        title,
        labels,
        save=None,
        sizex=20, 
        sizey=13, 
        show_labels=True,
        ):
    array = tensor.cpu().numpy()
    ax = plt.figure(figsize=(20, 13))
    plt.plot(array)
    plt.title(title)
    plt.grid()
    if show_labels:
        plt.xticks(np.arange(0, array.shape[0], 1), labels=labels)
        plt.xticks(rotation=90)
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

def pp_config(model):
    config = eval(model.cfg.__repr__()[len('HookedTransformerConfig:'):])

    # iterate over keys
    for k in config.keys():
        if 'd_' in k or 'n_' in k:
            print(f"{k:>30}\t: {config[k]}")



from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
from torch import Tensor
import einops
from pprint import pprint as pp
from tqdm import tqdm
import circuitsvis as cv
from transformers import LlamaForCausalLM, LlamaTokenizer
# t.set_grad_enabled(False)
from jaxtyping import Float, Int
import sys

if __name__ == "__main__":
    # device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # print(f"Device name: {t.cuda.get_device_name(device)}")
    # sys.path.append("/datadrive/madhur/mech_interp/TransformerLens/transformer_lens")
    
    device = t.device("cuda:0")
    from_pretrained_kwargs={
        "vocab_size": 54,
        "n_layer": 2,
        "n_embd": 128,
        "n_ctx": 40,
        "n_head": 1,
        "bos_token_id": 1, #tokenizer.bos_token_id, 
        "eos_token_id": 2, #tokenizer.eos_token_id,
        "pad_token_id": 3, #tokenizer.pad_token_id,
        # "act_fn": "gelu_new",
        # "# tokenizer_name": "gpt2",
        # "device": device,
        # "seed": 762568
    }

    model, _ = load_hooked_gpt2(device, from_pretrained_kwargs)
    breakpoint()