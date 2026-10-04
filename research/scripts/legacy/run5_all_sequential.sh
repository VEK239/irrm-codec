#!/bin/bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash research/scripts/legacy/run5a_hidden128.sh
bash research/scripts/legacy/run5b_blocks3.sh
bash research/scripts/legacy/run5c_hidden128_blocks3.sh
bash research/scripts/legacy/run5d_hidden96_blocks3.sh
