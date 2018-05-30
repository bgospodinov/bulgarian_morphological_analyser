if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import numpy as np

    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from helper import preprocess_dataset
    from config import config
    import argparse
    
    cols = list(config["DATASET"]["COLUMNS"].values())
    # UD --folder ../data/UD_Bulgarian-BTB --cols 1 2 4 --training bg_btb-ud-train.conllu --dev bg_btb-ud-dev.conllu --test bg_btb-ud-test.conllu
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="dataset folder path", type=str, 
                            default=os.path.join(os.path.pardir, "data", config["DATASET"]["FOLDER"]))
    parser.add_argument("--cols", help="list of column indices to read (order doesnt matter)", nargs='+', type=int, default=[0, 1, 2])
    
    for partition in config["DATASET"]["PARTITIONS"]:
        parser.add_argument("--{}".format(partition), 
                            help="{} file name".format(partition), 
                            type=str, 
                            default="{}.txt".format(partition))
    
    args = parser.parse_args()
    
    partition_dfs = {}
    partition_types = {}

    partition_dfs["pre_all_tokens"] = {}
    partition_dfs["all_tokens"] = {}

    # preprocessing
    for partition in config["DATASET"]["PARTITIONS"]:
        partition_df = pd.read_csv(os.path.join(args.folder, vars(args)[partition]), 
                    sep='\s+', 
                    names=cols,
                    usecols=args.cols)
        
        partition_dfs["pre_all_tokens"][partition] = partition_df
        partition_dfs["all_tokens"][partition] = preprocess_dataset(partition_df)
        
        types = {}
        for col in cols:
            types[col] = set(partition_dfs["all_tokens"][partition][col].unique())
        del col, partition_df

        partition_types[partition] = types
    del types, partition
    
    print(" Dataset statistics ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    print(("{:<20}" * 5).format("", "Word tokens", *["{} types".format(c).capitalize() for c in cols]))
    
    for partition in config["DATASET"]["PARTITIONS"]:
        pt = partition_types[partition]
        print(("{:<20}" * 5).format("{} set".format(partition), "{}/{}".format(len(partition_dfs["pre_all_tokens"][partition].index),\
                                                                               len(partition_dfs["all_tokens"][partition].index)), *[len(pt[c]) for c in cols]))
    del partition, pt
    print("\n")

    # cache a subset of the partition dataframe of ambiguous word tokens and tag tokens, unseen during training
    partition_dfs["unseen_word_tokens"] = {}
    partition_dfs["unseen_lemma_tokens"] = {}
    partition_dfs["unseen_tag_tokens"] = {}
    partition_dfs["ambiguous_word_tokens"] = {}

    print(" Types unseen during training ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    print(("{:<20}" * 5).format("", "", *["{} types".format(c).capitalize() for c in cols]))
    ptt = partition_types["training"]
    for partition in config["DATASET"]["PARTITIONS"]:
        if partition != "training":
            pt = partition_types[partition]
            print(("{:<20}" * 5).format("{} set".format(partition), "", *[len(pt[c] - ptt[c]) for c in cols]))

            # creating dataframes of unseen words during training for each partition of the dataset (i.e. valid and test sets)
            for c in cols:
                partition_dfs["unseen_{}_tokens".format(c)][partition] = partition_dfs["all_tokens"][partition]\
                        [partition_dfs["all_tokens"][partition][c].isin(pt[c] - ptt[c])]
    del partition, pt, ptt
    print("\n")

    # ambiguity stats
    print(" Ambiguous lemma types with more than one ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    print(("{:<20}" * 5).format("", "", "", *["{} type %".format(c).capitalize() for c in cols if c is not config["DATASET"]["COLUMNS"]["LEMMA"]]))
    for partition in config["DATASET"]["PARTITIONS"]:
        pdf = partition_dfs["all_tokens"][partition]
        print(("{:<20}" * 5).format("{} set".format(partition), "", "", *[pdf.groupby([config["DATASET"]["COLUMNS"]["LEMMA"]])[c].nunique().loc[lambda x: x > 1].size / pdf.groupby([config["DATASET"]["COLUMNS"]["LEMMA"]])[c].count().size for c in cols if c is not config["DATASET"]["COLUMNS"]["LEMMA"]]))
    del partition, pdf
    print("\n") 

    print(" Lemma types with most wordforms in training set ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    #print(partition_dfs["training"].groupby(["lemma"])["word"].nunique().loc[lambda x: x > 1].sort_values(ascending=False).head(20))
    print("\n") 
    
    print(" Lemma types with most tags in training set ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    #print(partition_dfs["training"].groupby(["lemma"])["tag"].nunique().loc[lambda x: x > 1].sort_values(ascending=False).head(20))
    print("\n") 
    # awk sanity check
    # awk 'BEGIN { cnt=0; } { if ( $2 == "голям" ) { cnt++; print tolower($1); } } END { print cnt; }' training.txt | sort | uniq -c | wc -l

       
    print(" Ambiguous word types with more than one ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    print(("{:<20}" * 5).format("", "", "", *["{} type %".format(c).capitalize() for c in cols if c is not config["DATASET"]["COLUMNS"]["WORD"]]))
    for partition in config["DATASET"]["PARTITIONS"]:
        pdf = partition_dfs["all_tokens"][partition]
        print(("{:<20}" * 5).format("{} set".format(partition), "", "", *[round(pdf.groupby([config["DATASET"]["COLUMNS"]["WORD"]])[c].nunique().loc[lambda x: x > 1].size / pdf.groupby([config["DATASET"]["COLUMNS"]["WORD"]])[c].count().size, 5) for c in cols if c is not config["DATASET"]["COLUMNS"]["WORD"]]))

        # awk 'BEGIN { cnt=0; } { if ( tolower($1) == "български" ) { cnt++; print tolower($1), $2, $3; } } END { print cnt; }' training.txt | sort | uniq -c

        # this line defines ambiguous word as having either more than one possible tag or one possible lemma
        #ambiguous_index = pdf.groupby([config["DATASET"]["COLUMNS"]["WORD"]]).nunique().apply(lambda x: x > 1).any(axis=1)
        ambiguous_index = pdf.groupby([config["DATASET"]["COLUMNS"]["WORD"]]).nunique()[config["DATASET"]["COLUMNS"]["LEMMA"]] > 1
        ambiguous_words = ambiguous_index.where(lambda x: x).dropna().index.tolist()
        partition_dfs["ambiguous_word_tokens"][partition] = pdf[pdf[config["DATASET"]["COLUMNS"]["WORD"]].isin(ambiguous_words)]
    del partition, pdf
    print("\n") 
    
    # naive baseline
    print(" Naive baseline ".upper().center(config["PPRINT"]["TITLE_LENGTH"], config["PPRINT"]["TITLE_CH"],))
    print("1) Assign each word form the most frequent lemma from the training set; otherwise assume every wordform is its own lemma.")
    print("2) Assign each word type the POS tag it was most frequently seen with in the training dataset; unknowns are wrong.")
    print("\n")
    
    # memoized baseline MFT tags and lemmata
    base_mem = {}        
    for col in cols:
        if col is not config["DATASET"]["COLUMNS"]["WORD"]:
            base_mem[col] = partition_dfs["pre_all_tokens"]["training"].groupby([config["DATASET"]["COLUMNS"]["WORD"], col]).size().to_frame("count")\
                .reset_index().sort_values([config["DATASET"]["COLUMNS"]["WORD"], "count"], ascending=[True, False])\
                .drop_duplicates(subset=config["DATASET"]["COLUMNS"]["WORD"]).iloc[:, 0:2]
    del col

    for scope in partition_dfs.keys():
        print(("{:<20}" * 6).format(scope.upper(), "", "Size", *["{} acc %".format(c).capitalize() for c in cols if c is not config["DATASET"]["COLUMNS"]["WORD"]], "Joint acc %"))

        for partition in config["DATASET"]["PARTITIONS"]:
            if partition != "training":
                pdf = partition_dfs[scope][partition]
                x_test = pdf[config["DATASET"]["COLUMNS"]["WORD"]]
                acc = []

                # initializes the mask that keeps record of which predictions are correct for every column of the dataset
                joint_mask = np.array([True] * pdf.shape[0])

                for c in cols:
                    if c is not config["DATASET"]["COLUMNS"]["WORD"]:
                        y_test = pdf[c].reset_index(drop=True)

                        y_pred = x_test.to_frame().set_index(config["DATASET"]["COLUMNS"]["WORD"])\
                            .join(base_mem[c].set_index(config["DATASET"]["COLUMNS"]["WORD"]), on=[config["DATASET"]["COLUMNS"]["WORD"]])\
                            .reset_index()

                        if c == config["DATASET"]["COLUMNS"]["LEMMA"]:
                            # interpolating lemmata
                            y_pred[c] = y_pred[c].fillna(y_pred[config["DATASET"]["COLUMNS"]["WORD"]])

                        pred_mask = (y_pred[c] == y_test)
                        acc.append(1 - pred_mask.value_counts(normalize=True).loc[False])
                        joint_mask &= pred_mask
                        #y_pred[pd.isnull(y_pred).any(axis=1)]

                # joint acc
                acc.append(joint_mask.sum() / joint_mask.size)

                print(("{:<20}" * 6).format("{} set".format(partition), "", pdf.shape[0], *acc))
        del partition, acc
        print("\n")