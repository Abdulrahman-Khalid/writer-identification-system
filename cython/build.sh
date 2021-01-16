#!/bin/bash
set -e

virtualenv .virtualenv;
source .virtualenv/bin/activate;

pip3 install -r requirements.txt;

python3 _cython_setup.py build_ext --inplace;