#!/bin/bash

conda activate stat214                # activates the stat214 environment

python clean.py                       # cleans and preprocesses the raw PECARN data
python models.py                      # trains and evaluates all three models

# optional: bootstrap stability analyses (slow, >10 min)
# python stability.py

conda deactivate                     # deactivates the conda environment
