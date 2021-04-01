# BERT-whitening

这是["Whitening Sentence Representations for Better Semantics and Faster Retrieval"](https://arxiv.org/abs/2103.15316)这篇论文的Pytorch实现版本.

BERT-whitening在工业界的文本语义相似度检索方面十分实用，whitening操作不仅提升了**无监督语义向量相似匹配**的效果，还能降低了向量维度，这有利于FAISS等向量检索引擎降低内存占用和提高检索速度。

本方法是苏剑林大神在其博客中首次提出的[\[1\]](https://kexue.fm/archives/8069)，本人看到后就联想到了PCA算法的降维特性，遂和苏神交流，再次感谢苏神带飞！

## 复现论文中的结果

### 准备工作
**下载数据集**:
```sh
$ cd data/
$ ./download_datasets.sh
$ cd ../
```
**下载预训练模型文件**:
```sh
$ cd model/
$ ./download_models.sh
$ cd ../
```

数据集和模型文件下载好以后，data/和model/目录如下:
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

### 复现BERT without whitening结果

```sh
$ python3 ./eval_without_whitening.py
```
试验结果如下:
|Model                   | STS-12  | STS-13 | STS-14 | STS-15 | STS-16 | SICK-R | STS-B  |
|:--                     | :--:    | :--:   | :--:   | :--:   | :--:   | :--:   | :--:   |
|BERTbase-cls            | 0.3062  | 0.2638 | 0.2765 | 0.3605 | 0.5180 | 0.4242 | 0.2029 |
|BERTbase-first_last_avg | 0.5785  | 0.6196 | 0.6250 | 0.7096 | 0.6979 | 0.6375 | 0.5904 |
|BERTlarge-cls           | 0.3240  | 0.2621 | 0.2629 | 0.3554 | 0.4439 | 0.4343 | 0.2675 |
|BERTlarge-first_last_avg| 0.5773  | 0.6116 | 0.6117 | 0.6806 | 0.7030 | 0.6034 | 0.5959 |

### 复现BERT with whitening(target)结果
```sh
$ python3 ./eval_with_whitening\(target\).py
```

试验结果如下:
|Model                            | STS-12  | STS-13 | STS-14 | STS-15 | STS-16 | SICK-R | STS-B  |
|:--                              | :--:    | :--:   | :--:   | :--:   | :--:   | :--:   | :--:   |
|BERTbase-whiten-256(target)      | 0.6390  | 0.7375 | 0.6909 | 0.7459 | 0.7442 | 0.6223 | 0.7143 |
|BERTlarge-whiten-384(target)     | 0.6435  | 0.7460 | 0.6964 | 0.7468 | 0.7594 | 0.6081 | 0.7247 |
|SBERTbase-nli-whiten-256(target) | 0.6912  | 0.7931 | 0.7805 | 0.8165 | 0.7958 | 0.7500 | 0.8074 |
|SBERTlarge-nli-whiten-384(target)| 0.7126  | 0.8061 | 0.7852 | 0.8201 | 0.8036 | 0.7402 | 0.8199 |

### 复现BERT with whitening(NLI)结果
```sh
$ python3 ./eval_with_whitening\(nli\).py
```

试验结果如下:
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


## 参考文献

[1] 苏剑林, [你可能不需要BERT-flow：一个线性变换媲美BERT-flow](https://kexue.fm/archives/8069), 2020.

[2] 苏剑林, [Keras版本BERT-whitening](https://github.com/bojone/BERT-whitening), 2020.
