# coding: utf-8
"""
使用target训练得到的whitening参数, 在下游任务测试.

@env: python3, pytorch>=1.7.1, transformers==4.2.0
@author: Weijie Liu
@date: 20/01/2020
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import scipy.stats
from all_utils import *
import senteval
import logging
from prettytable import PrettyTable


MAX_LENGTH = 64
BATCH_SIZE = 256
TEST_PATH = './data/'
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


MODEL_ZOOS = {

    'BERTbase-whiten(NLI)': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-base-uncased-first_last_avg-whiten(NLI).pkl',
        'n_components': 768,
    },

    'BERTbase-whiten-256(NLI)': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-base-uncased-first_last_avg-whiten(NLI).pkl',
        'n_components': 256,
    },

    'BERTlarge-whiten(NLI)': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-large-uncased-first_last_avg-whiten(NLI).pkl',
        'n_components': 1024,
    },

    'BERTlarge-whiten-384(NLI)': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-large-uncased-first_last_avg-whiten(NLI).pkl',
        'n_components': 384,
    },

    'SBERTbase-nli-whiten(NLI)': {
        'encoder': './model/bert-base-nli-mean-tokens',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-base-nli-mean-tokens-first_last_avg-whiten(NLI).pkl',
        'n_components': 768,
    },

    'SBERTbase-nli-whiten-256(NLI)': {
        'encoder': './model/bert-base-nli-mean-tokens',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-base-nli-mean-tokens-first_last_avg-whiten(NLI).pkl',
        'n_components': 256,
    },

    'SBERTlarge-nli-whiten(NLI)': {
        'encoder': './model/bert-large-nli-mean-tokens',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-large-nli-mean-tokens-first_last_avg-whiten(NLI).pkl',
        'n_components': 1024,
    },

    'SBERTlarge-nli-whiten-384(NLI)': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
        'whiten_file': './whiten/bert-large-nli-mean-tokens-first_last_avg-whiten(NLI).pkl',
        'n_components': 384,
    },

}


def prepare(params, samples):
    return None


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = []
    for sent in batch:
        vec = sent_to_vec(sent, params['tokenizer'], \
                params['encoder'], params['pooling'], MAX_LENGTH)
        embeddings.append(vec)
    embeddings = np.vstack(embeddings)
    embeddings = transform_and_normalize(embeddings, 
            kernel=params['whiten'][0],
            bias=params['whiten'][1]
        )  # whitening
    return embeddings


def run(model_name, test_path):

    model_config = MODEL_ZOOS[model_name]
    logging.info(f"{model_name} configs: {model_config}")

    tokenizer, encoder = build_model(model_config['encoder'])
    logging.info("Building {} tokenizer and model successfuly.".format(model_config['encoder']))

    kernel, bias = load_whiten(model_config['whiten_file'])
    kernel = kernel[:, :model_config['n_components']]
    logging.info("Loading kernel and bias from {}".format(model_config['whiten_file']))

    # Set params for senteval
    params_senteval = {
            'task_path': test_path,
            'usepytorch': True,
            'tokenizer': tokenizer,
            'encoder': encoder,
            'pooling': model_config['pooling'],
            'whiten': (kernel, bias),
            'batch_size': BATCH_SIZE
        }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
            'STSBenchmarkCosin',
            'SICKRelatednessCosin', 
        ]
    results = se.eval(transfer_tasks)
    
    # Show results
    table = PrettyTable(["Task", "Spearman"])
    for task in transfer_tasks:
        if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            metric = results[task]['all']['spearman']['wmean']
        elif task in ['SICKRelatednessCosin', 'STSBenchmarkCosin']:
            metric = results[task]['spearman']
        table.add_row([task, metric])
    logging.info(f"{model_name} results:\n" + str(table))


def run_all_model():

    for model_name in MODEL_ZOOS:
        run(model_name, TEST_PATH)


if __name__ == "__main__":
    # run('BERTbase-whiten-256(NLI)', TEST_PATH)
    run_all_model()


