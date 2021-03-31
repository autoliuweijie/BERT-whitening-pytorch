# coding: utf-8
"""
用AllNLI计算得到whitening的kernel和bias.

@env: python3, pytorch>=1.7.1, transformers==4.2.0
@author: Weijie Liu
@date: 20/01/2020
"""
import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import scipy.stats
import pickle
from all_utils import build_model, sents_to_vecs, compute_kernel_bias, save_whiten


NLI_PATH = './data/AllNLI.tsv'

MODEL_NAME_LIST = [
    './model/bert-base-uncased',
    './model/bert-large-uncased',
    './model/bert-base-nli-mean-tokens',
    './model/bert-large-nli-mean-tokens'
]

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

MAX_LENGTH = 64
OUTPUT_DIR = './whiten/'


def load_dataset(path):
    """
    loading AllNLI dataset.
    """
    senta_batch, sentb_batch = [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            items = line.strip().split('\t')
            senta, sentb = items[-3], items[-2]
            senta_batch.append(senta)
            sentb_batch.append(sentb)
    return senta_batch, sentb_batch


def main():

    a_sents_train, b_sents_train  = load_dataset(NLI_PATH)
    print("Loading {} training samples from {}".format(len(a_sents_train), NLI_PATH))

    for MODEL_NAME in MODEL_NAME_LIST:
        tokenizer, model = build_model(MODEL_NAME)
        print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

        print("Transfer sentences to BERT vectors.")
        a_vecs_train = sents_to_vecs(a_sents_train, tokenizer, model, POOLING, MAX_LENGTH)
        b_vecs_train = sents_to_vecs(b_sents_train, tokenizer, model, POOLING, MAX_LENGTH)

        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([a_vecs_train, b_vecs_train])

        model_name = MODEL_NAME.split('/')[-1]
        output_filename = f"{model_name}-{POOLING}-whiten(NLI).pkl"
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_whiten(output_path, kernel, bias)
        print("Save to {}".format(output_path))


if __name__ == "__main__":
    main()

