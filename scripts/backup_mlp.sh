#!/bin/bash
# Author: Bogomil Gospodinov
rsync -avzPucth -e 'ssh -o "ProxyCommand ssh -A s1312650@student.ssh.inf.ed.ac.uk nc %h %p"' \
--include="/models/***" --include="/data" --include="/data/input/***" --exclude="*" s1312650@mlp2:"~s1312650/msc_project/" "../"