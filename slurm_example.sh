#!/bin/bash

prefix=$1

export PATH="/media/compute/homes/pkenneweg/anaconda3/envs/test/bin:$PATH"
python3 ${prefix}/run_multiple.py 
