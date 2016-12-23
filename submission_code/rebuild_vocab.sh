#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./data/vocab_cut_.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ./data/vocab_rebuild.txt
