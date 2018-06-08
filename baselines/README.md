## How to train baselines
Run from root dir of the project. Only Lemming works with Cyrillic so far.
1. Preprocess files using `python -m data.preprocess_ud` \
e.g. `python -m data.preprocess_ud data/datasets/MorphoData-NewSplit/dev.txt > data/datasets/MorphoData-NewSplit/dev-pre.txt`\
	  This script removes most punctuation and lowercases all words, but preserves sentence boundaries. This step needs to be performed only once for each dataset and for all baselines like Lemming, Morfette etc.
3. Download mallet.jar, trove.jar, marmot.jar and place them in baselines/lemming/
4. Copy {dataset}_pre.txt file to baselines/lemming/data/{dataset}/
5. Make sure the dataset partition files from above are encoded in UTF-8 and with Unix EOL
6. Run `./baselines/lemming/lemming_btb.sh`
7. Predictions for dev set (by default) will be generated in baselines\lemming\predictions\btb under the name {lang}-{set}-pred.txt
8. Convert the predictions file to a format readable by our score script by running `./baselines/lemming/predictions/btb/convert_lemming_pred.awk {lang}-{set}-pred.txt > {lang}-{set}-pred-py.txt` and don't forget to manually encode in UTF-8 if necessary
9. To score it run `python -m analysis.score_prediction ../baselines/lemming/predictions/btb/bg-dev-pred-py.txt ../data/datasets/MorphoData-NewSplit/dev-pre.txt`