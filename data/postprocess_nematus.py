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
        line = re.sub(r"</?[^>]>|\s*", "", line)
        if tag_boundary not in line:
            buffer.write(line.rstrip() + '\t' + '\n')
            continue
        lemma_like, _, tag_like = line.rpartition(tag_boundary)
        buffer.write(lemma_like + '\t' + tag_like + '\n')
    buffer.seek(0)
    return buffer


def postprocess_sentence_to_sentence_with_missing_words(ground_stream, pred_stream):
    # expects original dev.txt file from before transformation
    ground_stream.seek(0)
    pred_stream.seek(0)
    ground_buffer = StringIO()
    pred_buffer = StringIO()

    for line in pred_stream:
        sent_sz = 0
        for row in ground_stream:
            col = row.strip()
            if not col:
                break
            ground_buffer.write(col + '\n')
            sent_sz += 1

        for word in line.split("<s>"):
            word = word.strip()
            if sent_sz == 0:
                break
            sent_sz -= 1
            pred_buffer.write(word + '\n')

        while sent_sz > 0:
            sent_sz -= 1
            pred_buffer.write('\n')

    return ground_buffer, pred_buffer


def postprocess_sentence_to_sentence(ground_stream, pred_stream):
    # expects dev_source file created for that model
    pred_stream.seek(0)
    ground_stream.seek(0)
    pred_buffer = StringIO()
    ground_buffer = StringIO()
    sent_sz = 0

    for ground_line in ground_stream:
        for word in ground_line.split("<s>"):
            word = word.strip()
            ground_buffer.write(word + '\n')
            sent_sz += 1

        pred_line = next(pred_stream)

        for word in pred_line.split("<s>"):
            if sent_sz == 0:
                break
            word = word.strip()
            pred_buffer.write(word + '\n')
            sent_sz -= 1

        while sent_sz > 0:
            sent_sz -= 1
            pred_buffer.write('\n')

    return ground_buffer, pred_buffer


if __name__ == "__main__":
    import sys
    import os
    from data.preprocess_ud import preprocess_dataset_for_train
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import config
    import logging
    cols = list(config["DATASET"]["COLUMNS"].values())

    level = logging.DEBUG
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')

    pred_path = sys.argv[1]
    ground_path = sys.argv[2]
    sentence_to_sentence_mode = "--sentence_to_sentence" in sys.argv

    pred_f = open(pred_path, mode="r", encoding='utf-8')

    if sentence_to_sentence_mode:
        pred_f_lc = 0
        ground_f_lc = 0

        for _ in pred_f:
            pred_f_lc += 1
        pred_f.seek(0)

        ground_f = open(ground_path, mode="r", encoding='utf-8')

        for _ in ground_f:
            ground_f_lc += 1
        ground_f.seek(0)

        print("{} ? {}".format(pred_f_lc, ground_f_lc), file=sys.stderr)
        assert pred_f_lc == ground_f_lc, "Number of predictions lines don't match with those in the ground file."
        ground_f, pred_f = postprocess_sentence_to_sentence(ground_f, pred_f)
        pred_f.seek(0)
        ground_f.seek(0)
        ground_f = postprocess_nematus(ground_f, config["TRANSFORM"]["DEFAULTS"]["TAG_BOUNDARY"])
        ground_df = pd.read_table(ground_f, header=None, names=['word'], usecols=[0])
    else:
        ground_df = pd.read_csv(ground_path, sep='\s+', names=cols, comment='#',
                                skip_blank_lines=(not sentence_to_sentence_mode))
        ground_df = preprocess_dataset_for_train(ground_df).reset_index()

    pred_f = postprocess_nematus(pred_f, config["TRANSFORM"]["DEFAULTS"]["TAG_BOUNDARY"])
    pred_df = pd.read_table(pred_f, header=None, names=['lemma', 'tag'])

    print("{} ? {}".format(ground_df.shape[0], pred_df.shape[0]), file=sys.stderr)
    assert ground_df.shape[0] == pred_df.shape[0], "Number of predicted rows doesn't match ground truth"

    joint_df = pd.concat([ground_df["word"], pred_df], axis=1, ignore_index=True)

    print("{} ? {}".format(ground_df.shape[0], joint_df.shape[0]), file=sys.stderr)
    assert ground_df.shape[0] == joint_df.shape[0], "More rows joined than necessary"

    output_buffer = StringIO()

    joint_df.to_string(output_buffer, index=False, header=False, na_rep=' ')

    output_buffer.seek(0)
    for line in output_buffer:
        print(line.strip())
