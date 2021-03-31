# coding: utf-8
import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import scipy.stats
import pickle


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(name):
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    model = model.to(DEVICE)
    return tokenizer, model


def sent_to_vec(sent, tokenizer, model, pooling, max_length):
    with torch.no_grad():
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=max_length)
        inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
        inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

        hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

        if pooling == 'first_last_avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'last_avg':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif pooling == 'last2avg':
            output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif pooling == 'cls':
            output_hidden_state = (hidden_states[-1])[:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(POOLING))

        vec = output_hidden_state.cpu().numpy()[0]
    return vec


def sents_to_vecs(sents, tokenizer, model, pooling, max_length, verbose=True):
    vecs = []
    if verbose:
        sents = tqdm(sents)
    for sent in sents:
        vec = sent_to_vec(sent, tokenizer, model, pooling, max_length)
        vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def calc_spearmanr_corr(x, y):
    return scipy.stats.spearmanr(x, y).correlation


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def save_whiten(path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }
    with open(path, 'wb') as f:
        pickle.dump(whiten, f)
    return path
    

def load_whiten(path):
    with open(path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
