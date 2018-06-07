## How to train baselines
Run from root dir of the project. Only Lemming works with Cyrillic so far.
1. Preprocess files using `python -m data.preprocess_ud` 
2. Download mallet.jar, trove.jar, marmot.jar and place them in baselines/lemming/
3. Copy {dataset}_pre.txt file to baselines/lemming/data/{dataset}/
4. Make sure the dataset partition files from above are encoded in UTF-8
5. Run `./baselines/lemming/lemming_btb.sh`
6. Predictions for dev set (by default) will be generated in baselines\lemming\predictions\btb under the name {lang}-{set}-pred.txt
7. Convert the predictions file to a format readable by our score script by running `./baselines/lemming/predictions/btb/convert_lemming_pred.awk {lang}-{set}-pred.txt > {lang}-{set}-pred-py.txt` and don't forget to manually encode in UTF-8 if necessary
8. To score it run `python -m analysis.score_prediction ../baselines/lemming/predictions/btb/bg-dev-pred-py.txt ../data/datasets/MorphoData-NewSplit/dev-pre.txt`