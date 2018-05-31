if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import config
    import argparse
    import pandas as pd
    from helper import preprocess_dataset

    cols = list(config["DATASET"]["COLUMNS"].values())
    #BTB ../baselines/lemming/predictions/btb/bg-dev-pred-py.txt ../data/MorphoData-NewSplit/dev.txt
    #UD ../baselines/lemming/predictions/ud/bg-dev-pred-py.txt ../baselines/lemming/data/UD_Bulgarian-BTB/bg-ud-dev.conllu.conv
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", help="training file name", type=str,
                        default=os.path.join(os.path.pardir, "data", config["DATASET"]["FOLDER"], "training.txt"))
    parser.add_argument("prediction", help="file name of predictions", type=str)
    parser.add_argument("ground", help="ground truth file name", type=str)
    parser.add_argument("--dataset_cols", help="list of column indices to read from dataset partitions (order doesnt matter)", nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--prediction_cols", help="list of column indices to read from prediction file (order doesnt matter)", nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()

    dfs = {}

    for partition in ["training", "prediction", "ground"]:
        pcols = args.dataset_cols if partition != "prediction" else args.prediction_cols
        dfs[partition] = preprocess_dataset(pd.read_csv(vars(args)[partition], sep='\s+', names=cols, usecols=pcols))

    assert dfs["prediction"].shape[0] == dfs["ground"].shape[0], "More rows predicted than necessary"

    def print_results(match):
        print(match["lemma_match"].value_counts(normalize=True))
        print(match["tag_match"].value_counts(normalize=True))
        print(match["joint_match"].value_counts(normalize=True))
        # prediction_match[prediction_match["joint_match"] == False]
        print("\n")

    print(" Prediction results for {} ".format(args.prediction).upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"]))
    print("\n")
    print("All tokens")
    prediction_match = dfs["prediction"].join(dfs["ground"], lsuffix='_prediction', rsuffix='_truth')
    prediction_match['lemma_match'] = prediction_match.apply(lambda row: row.lemma_prediction == row.lemma_truth,
                                       axis=1)
    prediction_match['tag_match'] = prediction_match.apply(lambda row: row.tag_prediction == row.tag_truth, axis=1)
    prediction_match['joint_match'] = prediction_match.apply(
        lambda row: row.tag_prediction == row.tag_truth and row.lemma_prediction == row.lemma_truth, axis=1)
    print_results(prediction_match)
    print("\n")

    print("Unseen tokens")
    training_words = dfs["training"]["word"].unique()
    prediction_match['seen'] = prediction_match.apply(lambda row: row.word_prediction in training_words, axis=1)
    print_results(prediction_match[prediction_match['seen'] == False])
    print("\n")

    print("Ambiguous tokens")
    ambiguous_index_ground = dfs["ground"].groupby("word").nunique()["lemma"] > 1
    ambiguous_index_training = dfs["training"].groupby("word").nunique()["lemma"] > 1
    joint_ambiguous_index = ambiguous_index_ground[ambiguous_index_ground == True].index.union(ambiguous_index_training[ambiguous_index_training == True].index)
    ambiguous_words = joint_ambiguous_index.tolist()
    prediction_match['ambiguous'] = prediction_match.apply(lambda row: row.word_prediction in ambiguous_words, axis=1)
    print_results(prediction_match[prediction_match['ambiguous'] == True])
    print("\n")

    print("Error analysis".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"]))
    print("Tagging errors")
    print(prediction_match[prediction_match["tag_match"] == False].groupby(["tag_prediction", "tag_truth"])["word_prediction"].count().sort_index(ascending=True).sort_values(ascending=False))
    print("\n")
    print("\n")

