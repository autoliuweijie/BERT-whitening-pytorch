#!/bin/bash

model_url="https://huggingface.co/bert-base-uncased"
model_name="bert-base-uncased"
if [ ! -r "$model_name" ]; then
    echo "Download ${model_name} from ${model_url}"
    git lfs clone ${model_url} 
else
    echo "${model_name} already exists."
fi

model_url="https://huggingface.co/bert-large-uncased"
model_name="bert-large-uncased"
if [ ! -r "$model_name" ]; then
    echo "Download ${model_name} from ${model_url}"
    git lfs clone ${model_url} 
else
    echo "${model_name} already exists."
fi

model_url="https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens"
model_name="bert-base-nli-mean-tokens"
if [ ! -r "$model_name" ]; then
    echo "Download ${model_name} from ${model_url}"
    git lfs clone ${model_url} 
else
    echo "${model_name} already exists."
fi

model_url="https://huggingface.co/sentence-transformers/bert-large-nli-mean-tokens"
model_name="bert-large-nli-mean-tokens"
if [ ! -r "$model_name" ]; then
    echo "Download ${model_name} from ${model_url}"
    git lfs clone ${model_url} 
else
    echo "${model_name} already exists."
fi
