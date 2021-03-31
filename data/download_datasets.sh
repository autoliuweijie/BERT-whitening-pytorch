#!/bin/bash

echo "Download AllNLI"
wget https://sbert.net/datasets/AllNLI.tsv.gz
gzip -d AllNLI.tsv.gz

echo "Download senteval datasets"
cd ./downstream/
./get_transfer_data.bash
