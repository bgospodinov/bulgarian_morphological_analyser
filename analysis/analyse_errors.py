#!/usr/bin/env python

__author__ = "Bogomil Gospodinov"
__email__ = "s1312650@sms.ed.ac.uk"
__status__ = "dev"

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    from spyder.utils.iofuncs import load_dictionary

    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import config
    import argparse
    
    cols = list(config["DATASET"]["COLUMNS"].values())
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="spydata file containing predictions matched against ground truth", type=str)
    parser.add_argument("--field", help="field to display in sentence", type=str, choices=["lemma", "tag"], default='lemma')
    args = parser.parse_args()

    spyder_dict = load_dictionary(args.input)[0]

    eof = False

    iterators = {}
    for name, prediction_match in spyder_dict['prediction_matches'].items():
        iterators[name] = prediction_match.iterrows()

    def print_table_row(name, sentence):
        print((" ".join(["{:15.15}"] * (len(sentence) + 1))).format(name, *sentence))

    while not eof:
        for iter_idx, (name, iterator) in enumerate(iterators.items()):
            if iter_idx == 0:
                sentence_truth = []
            sentence = []
            eof = True
            show = True

            for idx, row in iterator:
                eof = False
                pred = row["{}_prediction".format(args.field)]

                if not row["{}_match".format(args.field)]:
                    sentence.append(pred.upper())
                else:
                    sentence.append(pred)

                if iter_idx == 0:
                    sentence_truth.append(row["word_truth"])

                if row["eos"]:
                    break

            if iter_idx == 0:
                print_table_row("original_word", sentence_truth)

            print_table_row(name + "_lemma", sentence)
        print("\n")
