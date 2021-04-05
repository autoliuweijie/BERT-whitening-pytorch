#!/bin/bash

echo "Download AllNLI"
wget https://sbert.net/datasets/AllNLI.tsv.gz
gzip -d AllNLI.tsv.gz

echo "Download senteval datasets"
cd ./downstream/
./get_transfer_data.bash

echo "Download QQP datasets"
wget https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip
unzip QQP-clean.zip
rm QQP-clean.zip
