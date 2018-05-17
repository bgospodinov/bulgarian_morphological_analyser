import os
import sys
import pandas as pd

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
        partition_dfs[partition] = partition_df
        
        types = {}
        for col in cols:
            types[col] = partition_df[col].unique()
        del col

        partition_types[partition] = types
    del types, partition

    print(" Dataset token statistics ".upper().center(TITLE_LENGTH, TITLE_CH))
    print(("{:<20}" * 3).format("", "Word tokens", "Sentences"))
    
    for partition in config["DATASET"]["PARTITIONS"]:
        pt = partition_types[partition]
        print(("{:<20}" * 3).format("{} set".format(partition), 34324, 34324))
    del partition
    
    print("\n")
    
    print(" Dataset types statistics ".upper().center(TITLE_LENGTH, TITLE_CH))
    print(("{:<20}" * 4).format("", *["{} types".format(c).capitalize() for c in cols]))
    
    for partition in config["DATASET"]["PARTITIONS"]:
        pt = partition_types[partition]
        print(("{:<20}" * 4).format("{} set".format(partition), *[pt[c].size for c in cols]))
    del partition, pt