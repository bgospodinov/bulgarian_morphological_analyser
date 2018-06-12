#!/usr/bin/env python

""" postprocess_nematus.py: post-processing Nematus output for UD datasets for evaluation """
__author__ = "Bogomil Gospodinov"
__email__ = "s1312650@sms.ed.ac.uk"
__status__ = "dev"

import pandas as pd
from io import StringIO
import re

# example: python -m data.postprocess_nematus models/test-model/data/dev_hypothesis data/datasets/MorphoData-NewSplit/dev.txt > models/test-model/data/dev_prediction

def postprocess_nematus(stream, tag_boundary):
    """
    Converts a text stream of nematus predictions into a dataframe
    :param stream:
    :param tag_boundary:
    :return: dataframe with two columns: for lemmas and tags
    """
    stream.seek(0)
    buffer = StringIO()
    for line in stream:
        if tag_boundary not in line:
            continue
        lemma_like, _, tag_like = re.sub(r"</?[^>]>|\s*", "", line).rpartition(tag_boundary)
        buffer.write(lemma_like + '\t' + tag_like + '\n')
    buffer.seek(0)
    return pd.read_table(buffer, header=None, names=['lemma', 'tag'])


if __name__ == "__main__":
    import sys
    import os
    from data.preprocess_ud import preprocess_dataset_for_train
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import config
    cols = list(config["DATASET"]["COLUMNS"].values())
    pred_path = sys.argv[1]
    ground_path = sys.argv[2]

    pred_df = postprocess_nematus(open(pred_path, mode="r", encoding='utf-8'),
                                  config["TRANSFORM"]["DEFAULTS"]["TAG_BOUNDARY"])

    ground_df = preprocess_dataset_for_train(pd.read_csv(ground_path, sep='\s+', names=cols, comment='#'))\
        .reset_index()

    print("{} ? {}".format(ground_df.shape[0], pred_df.shape[0]), file=sys.stderr)
    assert ground_df.shape[0] == pred_df.shape[0], "More rows predicted than necessary"

    joint_df = pd.concat([ground_df["word"], pred_df], axis=1, ignore_index=True)

    print("{} ? {}".format(ground_df.shape[0], joint_df.shape[0]), file=sys.stderr)
    assert ground_df.shape[0] == joint_df.shape[0], "More rows joined than necessary"

    output_buffer = StringIO()

    joint_df.to_string(output_buffer, index=False, header=False, na_rep=' ')

    output_buffer.seek(0)
    for line in output_buffer:
        print(line.strip())
