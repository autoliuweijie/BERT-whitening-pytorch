# BERT-whitening

This is the Pytorch implementation of ["Whitening Sentence Representations for Better Semantics and Faster Retrieval"](https://arxiv.org/abs/2103.15316).

BERT-whitening is very practical in text semantic search, in which the whitening operation not only improves the performance of **unsupervised semantic vector matching**, but also reduces the vector dimension, which is beneficial to reduce memory usage and improve retrieval efficiency for vector search engines, e.g., FAISS.

This method was first proposed by Jianlin Su in his blog[\[1\]](https://kexue.fm/archives/8069). 

## Reproduce the experimental results

### Preparation
Download datasets
```sh
$ cd data/
$ ./download_datasets.sh
$ cd ../
```
Download models
```sh
$ cd model/
$ ./download_models.sh
$ cd ../
```

After the datasets and models files are downloaded, the ``data/`` and ``model/`` directories are as follows:
```
├── data
│   ├── AllNLI.tsv
│   ├── download_datasets.sh
│   └── downstream
│       ├── COCO
│       ├── CR
│       ├── get_transfer_data.bash
│       ├── MPQA
│       ├── MR
│       ├── MRPC
│       ├── SICK
│       ├── SNLI
│       ├── SST
│       ├── STS
│       ├── SUBJ
│       ├── tokenizer.sed
│       └── TREC
├── model
│   ├── bert-base-nli-mean-tokens
│   ├── bert-base-uncased
│   ├── bert-large-nli-mean-tokens
│   ├── bert-large-uncased
│   └── download_models.sh

```

### BERT without whitening

```sh
$ python3 ./eval_without_whitening.py
```
Results:
|Model                   | STS-12  | STS-13 | STS-14 | STS-15 | STS-16 | SICK-R | STS-B  |
|:--                     | :--:    | :--:   | :--:   | :--:   | :--:   | :--:   | :--:   |
|BERTbase-cls            | 0.3062  | 0.2638 | 0.2765 | 0.3605 | 0.5180 | 0.4242 | 0.2029 |
|BERTbase-first_last_avg | 0.5785  | 0.6196 | 0.6250 | 0.7096 | 0.6979 | 0.6375 | 0.5904 |
|BERTlarge-cls           | 0.3240  | 0.2621 | 0.2629 | 0.3554 | 0.4439 | 0.4343 | 0.2675 |
|BERTlarge-first_last_avg| 0.5773  | 0.6116 | 0.6117 | 0.6806 | 0.7030 | 0.6034 | 0.5959 |

### BERT with whitening(target)
```sh
$ python3 ./eval_with_whitening\(target\).py
```

Results:
|Model                            | STS-12  | STS-13 | STS-14 | STS-15 | STS-16 | SICK-R | STS-B  |
|:--                              | :--:    | :--:   | :--:   | :--:   | :--:   | :--:   | :--:   |
|BERTbase-whiten-256(target)      | 0.6390  | 0.7375 | 0.6909 | 0.7459 | 0.7442 | 0.6223 | 0.7143 |
|BERTlarge-whiten-384(target)     | 0.6435  | 0.7460 | 0.6964 | 0.7468 | 0.7594 | 0.6081 | 0.7247 |
|SBERTbase-nli-whiten-256(target) | 0.6912  | 0.7931 | 0.7805 | 0.8165 | 0.7958 | 0.7500 | 0.8074 |
|SBERTlarge-nli-whiten-384(target)| 0.7126  | 0.8061 | 0.7852 | 0.8201 | 0.8036 | 0.7402 | 0.8199 |

### BERT with whitening(NLI)
```sh
$ python3 ./eval_with_whitening\(nli\).py
```

Results:
|Model                            | STS-12  | STS-13 | STS-14 | STS-15 | STS-16 | SICK-R | STS-B  |
|:--                              | :--:    | :--:   | :--:   | :--:   | :--:   | :--:   | :--:   |
|BERTbase-whiten(nli)             | 0.6169  | 0.6571 | 0.6605 | 0.7516 | 0.7320 | 0.6829 | 0.6365 |
|BERTbase-whiten-256(nli)         | 0.6148  | 0.6672 | 0.6622 | 0.7483 | 0.7222 | 0.6757 | 0.6496 |
|BERTlarge-whiten(nli)            | 0.6254  | 0.6737 | 0.6715 | 0.7503 | 0.7636 | 0.6865 | 0.6250 |
|BERTlarge-whiten-348(nli)        | 0.6231  | 0.6784 | 0.6701 | 0.7548 | 0.7546 | 0.6866 | 0.6381 |
|SBERTbase-nli-whiten(nli)        | 0.6868  | 0.7646 | 0.7626 | 0.8230 | 0.7964 | 0.7896 | 0.7653 |
|SBERTbase-nli-whiten-256(nli)    | 0.6891  | 0.7703 | 0.7658 | 0.8229 | 0.7828 | 0.7880 | 0.7678 |
|SBERTlarge-nli-whiten(nli)       | 0.7074  | 0.7756 | 0.7720 | 0.8285 | 0.8080 | 0.7910 | 0.7589 |
|SBERTlarge-nli-whiten-384(nli)   | 0.7123  | 0.7893 | 0.7790 | 0.8355 | 0.8057 | 0.8037 | 0.7689 |


## References

[1] 苏剑林, [你可能不需要BERT-flow：一个线性变换媲美BERT-flow](https://kexue.fm/archives/8069), 2020.

[2] 苏剑林, [Keras版本BERT-whitening](https://github.com/bojone/BERT-whitening), 2020.
