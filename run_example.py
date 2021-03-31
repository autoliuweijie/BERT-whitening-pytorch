# coding: utf-8
"""
BERT-PCA: 在BERT embedding后面接一个PCA, 得到句子的句向量表示, 
用于计算两段文本相似度.

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

TRAIN_PATH = './data/downstream/STS/STSBenchmark/sts-train.csv'
DEV_PATH = './data/downstream/STS/STSBenchmark/sts-dev.csv'
TEST_PATH = './data/downstream/STS/STSBenchmark/sts-test.csv'

MODEL_NAME = './model/bert-base-uncased' # 本地模型文件
# MODEL_NAME = './model/bert-large-uncased' # 本地模型文件
# MODEL_NAME = './model/sbert-base-uncased-nli' # 本地模型文件
# MODEL_NAME = './model/sbert-large-uncased-nli' # 本地模型文件

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

USE_WHITENING = True
N_COMPONENTS = 256
MAX_LENGTH = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(path, test_or_train):
    """
    loading training or testing dataset.
    """
    senta_batch, sentb_batch, scores_batch = [], [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            items = line.strip().split('\t')
            if test_or_train == 'train':
                senta, sentb, score = items[-2], items[-1], float(items[-3])
            elif test_or_train in ['dev', 'test']:
                senta, sentb, score = items[-2], items[-1], float(items[-3])
            else:
                raise Exception("{} error".format(test_or_train))
            senta_batch.append(senta)
            sentb_batch.append(sentb)
            scores_batch.append(score)
    return senta_batch, sentb_batch, scores_batch


def build_model(name):
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    model = model.to(DEVICE)
    return tokenizer, model


def sents_to_vecs(sents, tokenizer, model):
    vecs = []
    with torch.no_grad():
        for sent in tqdm(sents):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'last_avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif POOLING == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling {}".format(POOLING))

            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs


def calc_spearmanr_corr(x, y):
    return scipy.stats.spearmanr(x, y).correlation


def compute_kernel_bias(vecs, n_components):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


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


def main():

    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_WHITENING}-{N_COMPONENTS}.")

    a_sents_train, b_sents_train, scores_train = load_dataset(TRAIN_PATH, 'train')
    a_sents_test, b_sents_test, scores_test = load_dataset(TEST_PATH, 'test')
    a_sents_dev, b_sents_dev, scores_dev = load_dataset(DEV_PATH, 'dev')
    print("Loading {} training samples from {}".format(len(scores_train), TRAIN_PATH))
    print("Loading {} developing samples from {}".format(len(scores_dev), DEV_PATH))
    print("Loading {} testing samples from {}".format(len(scores_test), TEST_PATH))

    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

    print("Transfer sentences to BERT vectors.")
    a_vecs_test = sents_to_vecs(a_sents_test, tokenizer, model)
    b_vecs_test = sents_to_vecs(b_sents_test, tokenizer, model)

    if USE_WHITENING:
        a_vecs_train = sents_to_vecs(a_sents_train, tokenizer, model)
        b_vecs_train = sents_to_vecs(b_sents_train, tokenizer, model)
        a_vecs_dev = sents_to_vecs(a_sents_dev, tokenizer, model)
        b_vecs_dev = sents_to_vecs(b_sents_dev, tokenizer, model)

        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([
            a_vecs_train, b_vecs_train, a_vecs_dev, b_vecs_dev, a_vecs_test, b_vecs_test
        ], n_components=N_COMPONENTS)

        a_vecs_test = transform_and_normalize(a_vecs_test, kernel, bias)
        b_vecs_test = transform_and_normalize(b_vecs_test, kernel, bias)
    else:
        a_vecs_test = normalize(a_vecs_test)
        b_vecs_test = normalize(b_vecs_test)


    print("Results:")
    test_sims = (a_vecs_test * b_vecs_test).sum(axis=1)
    print(u'Spearmanr corr in Testing set：%s' % calc_spearmanr_corr(scores_test, test_sims))


if __name__ == "__main__":
    main()

