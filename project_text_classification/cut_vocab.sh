#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./newData/vocab_cut_build_.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[0]\s" | cut -d' ' -f2 > ./newData/vocab_cut_build_cut_.txt
