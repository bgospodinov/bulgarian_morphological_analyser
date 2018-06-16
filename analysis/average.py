#!/usr/bin/env python
import sys
import pandas as pd

scores = pd.read_csv(sys.argv[1], sep=",", header=None).astype('float64')
print(", ".join([str(round(score, 5)) for score in scores.mean(axis=0).tolist()]))

