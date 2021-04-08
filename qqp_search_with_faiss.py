# coding: utf-8
"""
Test search performance with Approximate Nearest Neighbor (ANN), e.g., FAISS.

As dataset, we use the Quora Duplicate Questions dataset, which contains about
400k questions: https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs.

@author: jagerliu
"""
import os
import faiss
import pickle
import numpy as np
import collections
from all_utils import build_model, sent_to_vec, http_get, load_whiten
from all_utils import compute_kernel_bias, transform_and_normalize, normalize
from all_utils import get_size
from tqdm import tqdm
import time


MODEL_ZOOS = {

    'BERTbase-first_last_avg': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'first_last_avg',
    },

    'BERTbase-cls': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'cls',
    },

    'BERTbase-whiten(target)': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'first_last_avg',
        'n_components': 768,
    },

    'BERTbase-whiten-256(target)': {
        'encoder': './model/bert-base-uncased',
        'pooling': 'first_last_avg',
        'n_components': 256,
    },

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

    'BERTlarge-first_last_avg': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
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

    'BERTlarge-whiten(target)': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
        'n_components': 1024,
    },

    'BERTlarge-whiten-384(target)': {
        'encoder': './model/bert-large-uncased',
        'pooling': 'first_last_avg',
        'n_components': 384,
    },
}


# Model config
# model_name ='BERTbase-whiten(NLI)'
# model_name ='BERTbase-whiten-256(NLI)'
# model_name ='BERTbase-whiten-256(target)'
# model_name ='BERTbase-whiten(target)'
# model_name ='BERTbase-first_last_avg'                                                                                                                                                                                                                                                       
# model_name ='BERTlarge-whiten(NLI)'
# model_name ='BERTlarge-whiten-384(NLI)'
# model_name ='BERTlarge-whiten(target)'
# model_name ='BERTlarge-whiten-384(target)'
model_name ='BERTlarge-first_last_avg'
max_length = 64


# Dataset config
data_url = 'http://www.weijieliu.com/download/datasets/Quora_Duplicate_Questions/quora_duplicate_questions.tsv'
dataset_path = "data/quora_duplicate_questions.tsv"
max_corpus_size = 100000
embedding_cache_path = 'data/quora-embeddings-{}-size-{}.pkl'.format(
        MODEL_ZOOS[model_name]['encoder'].replace('/', '_'), max_corpus_size)


# FAISS config
top_k_hits = 10
n_clusters = 1024
nprobe = 5


model_config = MODEL_ZOOS[model_name]
print(f"{model_name} configs: {model_config}")
tokenizer, encoder = build_model(model_config['encoder'])
print("Building {} tokenizer and model successfuly.".format(model_config['encoder']))
ques_to_vec = lambda ques: \
    sent_to_vec(ques, tokenizer, encoder, model_config['pooling'], max_length)


def load_qqp_vecs():

    # Get sentence vectors
    if not os.path.exists(embedding_cache_path):
        if not os.path.exists(dataset_path):
            print("Download dataset")
            http_get(data_url, dataset_path)
        print("Extract sentence vectors")
        qid1_to_vec = collections.OrderedDict()  # {qid: vec}
        qid2_to_vec = collections.OrderedDict()  # {qid: vec}
        qid1_gold_trues = collections.OrderedDict()  # {qid1: [qid2, qid2, ...]}
        with open(dataset_path, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                if i == 0:
                    continue
                _, qid1, qid2, ques1, ques2, is_dup = line.strip().split('\t')
                qid1, qid2 = int(qid1), int(qid2)
                vec1 = ques_to_vec(ques1)
                vec2 = ques_to_vec(ques2)
                qid1_to_vec[qid1] = vec1
                qid2_to_vec[qid2] = vec2
                if is_dup == '1':
                    if qid1 in qid1_gold_trues.keys():
                        qid1_gold_trues[qid1].append(qid2)
                    else:
                        qid1_gold_trues[qid1] = [qid2]
                else:
                    if qid1 not in qid1_gold_trues.keys():
                        qid1_gold_trues[qid1] = []
                if i > max_corpus_size:
                    break
        print("Saving vectos to {}".format(embedding_cache_path))
        with open(embedding_cache_path, 'wb') as fout:
            dump_obj = (qid1_to_vec, qid2_to_vec, qid1_gold_trues)
            pickle.dump(dump_obj, fout)
    else:
        print("Loading vectos from {}".format(embedding_cache_path))
        with open(embedding_cache_path, 'rb') as fin:
            dump_obj = pickle.load(fin)
            qid1_to_vec, qid2_to_vec, qid1_gold_trues = dump_obj

    # Format sentence vectors
    vecs1 = np.vstack([v for _, v in qid1_to_vec.items()])
    qids1 = np.array([i for i, _ in qid1_to_vec.items()])
    vecs2 = np.vstack([v for _, v in qid2_to_vec.items()])
    qids2= np.array([i for i, _ in qid2_to_vec.items()])

    return qids1, vecs1, qids2, vecs2, qid1_gold_trues


def main():

    # Load vectors
    qids1, vecs1, qids2, vecs2, qid1_gold_trues = load_qqp_vecs()

    # Normalize or Whitening vectors
    if "whiten" in model_name:
        print("Whitening vectors")
        if 'whiten_file' in model_config.keys():
            print("Use whiten file : {}".format(model_config['whiten_file']))
            kernel, bias = load_whiten(model_config['whiten_file'])
        else:
            print("Compulate kernel and bias")
            kernel, bias = compute_kernel_bias([vecs1, vecs2])
        kernel = kernel[:, :model_config['n_components']]
        vecs1 = transform_and_normalize(vecs1, kernel, bias)
        vecs2 = transform_and_normalize(vecs2, kernel, bias)
    else:
        print("Normalze vectors")
        vecs1 = normalize(vecs1)
        vecs2 = normalize(vecs2)
    vecs1 = vecs1.astype('float32')
    vecs2 = vecs2.astype('float32')
    print("Query num:", len(vecs1))
    print("Docs num:", len(vecs2))
    vecs2_memory_size = get_size(vecs2)
    print("Vecs2 Memory Size: {} GB".format(vecs2_memory_size / 1073741824.0))

    # Create Index
    embedding_size = vecs2.shape[1]    #Size of embeddings
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe
    index.train(vecs2)
    index.add_with_ids(vecs2, qids2)

    # Search
    start = time.time()
    scores, retids = index.search(vecs1, top_k_hits)
    end = time.time()
    print("Average search time: {} ms".format((end - start) * 1000.0 / len(vecs1)))

    # Calculate MRR
    mrr_list = []
    for i in range(len(vecs1)):
        golds = qid1_gold_trues[qids1[i]]
        score = scores[i]
        if len(golds) > 0:
            for rank, retid in enumerate(retids[i]):
                if retid in golds:
                    mrr_tmp = 1 / (rank + 1.0)
                    break
                mrr_tmp = 0.0
            mrr_list.append(mrr_tmp)

            # print("RetID:", retids[i])
            # print("Gold: ", golds)
            # print("Score: ", score)
    mrr = np.mean(mrr_list)
    print("MRR: ", mrr)


if __name__ == "__main__":
    main()

