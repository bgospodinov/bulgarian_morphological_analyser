#!/usr/bin/env python

""" preprocess_ud.py: pre-processing of UD datasets for training and evaluation """
__author__ = "Bogomil Gospodinov"
__email__ = "s1312650@sms.ed.ac.uk"
__status__ = "dev"

import pandas as pd
import re

def preprocess_dataset_for_train(df):
    """
    Exludes punctuation (except parenthesis and hyphens), lowercases every entity,
    and preserves sentence boundaries (newlines) is important for Lemming parsing context properly.
    This function leaves in some entities which are useful for context but that ultimately we don't want to evaluate on.
    Use postprocess_dataset to remove those.
    :param df:
    :return:
    """
    # sanity check
    # awk -F "[[:space:]]+" 'BEGIN{ cnt=0; } $1 ~ /[[:alpha:]]+/ { cnt+=1; } END{ print cnt; }' training.txt
    df = df.apply(lambda x: x.astype(str).str.lower() if not pd.isna(x).any() else x, axis=1)
    # Bulgarian BTB-UD specific preprocessing
    df["tag"] = df["tag"].apply(lambda x: re.sub(r"(?:\+[0-9]+|\".*|;.*)+$", "", x) if not pd.isna(x) else x)
    # ignore corrupted entries where only the word is null but not the tag
    df = df.loc[~(pd.isna(df["word"]) & ~pd.isna(df["tag"]))]
    # ignore punctuation by tag and by string but keep nans that represent new lines
    df = df[~df["word"].str.contains(r"^(?:\"|!|'|,|\.|\?|:|;|\||\-|\+|\(|\)|%|\*|/|&|=|>|<|_)", na=False)]
    df = df[((df["tag"] != "punct") & (df["lemma"] != "punct")) | pd.isna(df["word"])]
    return df


def preprocess_dataset_for_eval(df, prediction=False):
    """
    Excludes numerals and all non-cyrillic characters, as well as all proper nouns and other names.
    Removes all entities that we don't want to include in the final evaluation score.
    :param df:
    :return:
    """
    if prediction:
        word_col = "word_truth"
        tag_col = "tag_truth"
    else:
        word_col = "word"
        tag_col = "tag"
    return df[(df[word_col].str.match(r'[\u0400-\u04FF]+[\u0400-\u04FF\-()]*')) &\
              (~(df[tag_col].str.contains("^(?:Np|H|punct).*$", case=False).fillna(False)))]


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import config
    from io import StringIO
    cols = list(config["DATASET"]["COLUMNS"].values())
    dataset_path = sys.argv[1]
    df = pd.read_csv(dataset_path, sep='\s+', names=cols, skip_blank_lines=False, comment='#')
    df = preprocess_dataset_for_train(df)
    buffer = StringIO()
    df.to_string(buffer, index=False, header=False, na_rep=' ')
    buffer.seek(0)
    for line in buffer:
        print(line.strip())