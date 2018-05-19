import os
import sys
import pandas as pd

pd.set_option('display.max_rows', 50) 
pd.set_option('display.max_columns', None)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config

if __name__ == "__main__":    
    import argparse
    TITLE_LENGTH = 90
    TITLE_CH = "="
    cols = list(config["DATASET"]["COLUMNS"].values())
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="dataset folder path", type=str, 
                            default=os.path.join(os.path.pardir, "data", config["DATASET"]["FOLDER"]))
    
    for partition in config["DATASET"]["PARTITIONS"]:
        parser.add_argument("--{}".format(partition), 
                            help="{} file name".format(partition), 
                            type=str, 
                            default="{}.txt".format(partition))
    
    args = parser.parse_args()
    
    partition_dfs = {}
    partition_types = {}
    
    for partition in config["DATASET"]["PARTITIONS"]:
        partition_df = pd.read_csv(os.path.join(args.folder, vars(args)[partition]), 
                    sep='\s+', 
                    names=cols)
        partition_df["word"] = partition_df["word"].str.lower()
        partition_dfs[partition] = partition_df
        
        types = {}
        for col in cols:
            types[col] = set(partition_df[col].unique())
        del col, partition_df

        partition_types[partition] = types
    del types, partition
    
    print(" Dataset statistics ".upper().center(TITLE_LENGTH, TITLE_CH))
    print(("{:<20}" * 5).format("", "Word tokens", *["{} types".format(c).capitalize() for c in cols]))
    
    for partition in config["DATASET"]["PARTITIONS"]:
        pt = partition_types[partition]
        print(("{:<20}" * 5).format("{} set".format(partition), len(partition_dfs[partition].index), *[len(pt[c]) for c in cols]))
    del partition, pt
    print("\n")
    
    print(" Types unseen during training ".upper().center(TITLE_LENGTH, TITLE_CH))
    print(("{:<20}" * 5).format("", "", *["{} types".format(c).capitalize() for c in cols]))
    ptt = partition_types["training"]
    for partition in config["DATASET"]["PARTITIONS"]:
        if partition != "training":
            pt = partition_types[partition]
            print(("{:<20}" * 5).format("{} set".format(partition), "", *[len(pt[c] - ptt[c]) for c in cols]))
    del partition, pt, ptt     
    print("\n")    
    
    # ambiguity stats
    print(" Ambiguous lemma types with more than one ".upper().center(TITLE_LENGTH, TITLE_CH))
    print(("{:<20}" * 5).format("", "", "", *["{} type %".format(c).capitalize() for c in cols if c is not config["DATASET"]["COLUMNS"]["LEMMA"]]))
    for partition in config["DATASET"]["PARTITIONS"]:
        pdf = partition_dfs[partition]
        print(("{:<20}" * 5).format("{} set".format(partition), "", "", *[pdf.groupby([config["DATASET"]["COLUMNS"]["LEMMA"]])[c].nunique().loc[lambda x: x > 1].size / pdf.groupby([config["DATASET"]["COLUMNS"]["LEMMA"]])[c].count().size for c in cols if c is not config["DATASET"]["COLUMNS"]["LEMMA"]]))
    del partition, pdf
    print("\n") 

    
    print(" Lemma types with most wordforms in training set ".upper().center(TITLE_LENGTH, TITLE_CH))
    #print(partition_dfs["training"].groupby(["lemma"])["word"].nunique().loc[lambda x: x > 1].sort_values(ascending=False).head(20))
    print("\n") 
    
    print(" Lemma types with most tags in training set ".upper().center(TITLE_LENGTH, TITLE_CH))
    #print(partition_dfs["training"].groupby(["lemma"])["tag"].nunique().loc[lambda x: x > 1].sort_values(ascending=False).head(20))
    print("\n") 
    # awk sanity check
    # awk 'BEGIN { cnt=0; } { if ( $2 == "голям" ) { cnt++; print tolower($1); } } END { print cnt; }' training.txt | sort | uniq -c | wc -l

       
    print(" Ambiguous word types with more than one ".upper().center(TITLE_LENGTH, TITLE_CH))
    print(("{:<20}" * 5).format("", "", "", *["{} type %".format(c).capitalize() for c in cols if c is not config["DATASET"]["COLUMNS"]["WORD"]]))
    for partition in config["DATASET"]["PARTITIONS"]:
        pdf = partition_dfs[partition]
        print(("{:<20}" * 5).format("{} set".format(partition), "", "", *[round(pdf.groupby([config["DATASET"]["COLUMNS"]["WORD"]])[c].nunique().loc[lambda x: x > 1].size / pdf.groupby([config["DATASET"]["COLUMNS"]["WORD"]])[c].count().size, 5) for c in cols if c is not config["DATASET"]["COLUMNS"]["WORD"]]))
    del partition, pdf
    print("\n") 
    
    # naive baseline
    print(" Naive baseline ".upper().center(TITLE_LENGTH, TITLE_CH))
    print("1) Assign each word form the most frequent lemma from the training set; otherwise assume every wordform is its own lemma.")
    print("2) Assign each word type the POS tag it was most frequently seen with in the training dataset; unknowns are wrong.")
    print("\n")
    
    # memoized baseline MFT tags and lemmata
    base_mem = {}        
    for col in cols:
        if col is not config["DATASET"]["COLUMNS"]["WORD"]:
            base_mem[col] = partition_dfs["training"].groupby(["word", col]).size().to_frame("count")\
                .reset_index().sort_values(["word", "count"], ascending=[True, False])\
                .drop_duplicates(subset="word").iloc[:, 0:2]
    del col
    
    print(("{:<20}" * 5).format("", "", "", *["{} acc %".format(c).capitalize() for c in cols if c is not config["DATASET"]["COLUMNS"]["WORD"]]))
    
    for partition in config["DATASET"]["PARTITIONS"]:
        if partition != "training":
            pdf = partition_dfs[partition]
            x_test = pdf["word"]
            acc = [] 
            
            for c in cols:
                if c is not config["DATASET"]["COLUMNS"]["WORD"]:
                    y_test = pdf[c]
                    
                    y_pred = x_test.to_frame().set_index('word')\
                        .join(base_mem[c].set_index('word'), on=["word"])\
                        .reset_index()
                    
                    if c == config["DATASET"]["COLUMNS"]["LEMMA"]:
                        # interpolating lemmata
                        y_pred[c] = y_pred[c].fillna(y_pred['word'])
                    
                    
                    acc.append((y_pred[c] == y_test).value_counts(normalize=True).iloc[0])
                    #y_pred[pd.isnull(y_pred).any(axis=1)]
            
            print(("{:<20}" * 5).format("{} set".format(partition), "", "", *acc))
    del partition