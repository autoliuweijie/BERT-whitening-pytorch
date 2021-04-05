# coding: utf-8
import os
import sys
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import scipy.stats
import pickle
import requests


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
    return normalize(vecs)


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


import inspect

def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size
