#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./data/vocab_rebuild.txt | sed "s/^\s\+//g" | sort -rn | cut -d' ' -f2 > ./data/vocab_rebuild_recut.txt
