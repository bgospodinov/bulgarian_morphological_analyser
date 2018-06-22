#!/bin/bash
# Author: Bogomil Gospodinov
rsync -avzPucth -e 'ssh -v -o "ProxyCommand ssh -A -v s1312650@student.ssh.inf.ed.ac.uk nc %h %p"' \
--max-size=50m \
--include="/models/***" --include="/data" --include="/data/input/***" --include="/logs/***" --exclude="*" \
s1312650@mlp1:"~s1312650/msc_project/" "./"