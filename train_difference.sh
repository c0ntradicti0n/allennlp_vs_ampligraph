rm -r ./output/$1
allennlp train \
               --include-package nym_embeddings.nym_embeddings \
               --include-package nym_embeddings.synset_indexer \
               $1 -s ./output/$1/

