#!/bin/bash
# Author: Bogomil Gospodinov
rsync -avzPucth -e 'ssh -o "ProxyCommand ssh -A s1312650@student.ssh.inf.ed.ac.uk nc %h %p"' \
--max-size=50m \
--include="/models/***" --include="/data" --include="/data/input/***" --include="/scripts" --include="/scripts/logs/***" --exclude="*" \
s1312650@mlp2:"~s1312650/msc_project/" "./"