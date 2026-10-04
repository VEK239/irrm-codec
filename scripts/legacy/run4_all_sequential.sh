#!/bin/bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/legacy/run4a_individual_maxlen_maxtokenlen9_anchored.sh
bash scripts/legacy/run4b_individual_maxlen_maxtokenlen9_anchored_dropout.sh