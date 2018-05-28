def preprocess_dataset(df):
    # awk -F "[[:space:]]+" 'BEGIN{ cnt=0; } $1 ~ /[[:alpha:]]+/ { cnt+=1; } END{ print cnt; }' training.txt
    # exlude punctuation, numerals and all non-cyrillic characters
    df["word"] = df["word"].str.lower()
    df["tag"] = df["tag"].str.lower()
    return df[(df["word"].str.match(r'[\u0400-\u04FF]+[\u0400-\u04FF\-()]*')) &\
                             (df["tag"] != "punct") &\
                             (df["lemma"] != "punct")]
