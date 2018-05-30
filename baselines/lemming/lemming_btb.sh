#Partial credit to Tom Bergmannis
#read about these in lemmings web page
options=use-morph=false,use-perceptron=false
options=$options,use-mallet=false,offline-feature-extraction=false
options=$options,tag-dependent=true,seed=12345

#language extension
lang=bg
set=dev

mkdir -p models

#format of the train file as below
#можем \s+ мога \s+ VERB
# empty line used as a sentence separator

#train POS model
#java -d64 -Xmx10G -cp marmot.jar marmot.morph.cmd.Trainer -tag-morph false -train-file form-index=0,tag-index=2,data/MorphoData-NewSplit/${lang}-train.txt.conv -model-file models/btb/${lang}.marmot 

#train lemmatizer
#java -d64 -Xmx10G -cp "marmot.jar;mallet.jar;trove.jar" lemming.lemma.cmd.Trainer lemming.lemma.ranker.RankerTrainer $options models/btb/${lang}.lemming form-index=0,lemma-index=1,data/MorphoData-NewSplit/${lang}-train.txt.conv  

#make predictions 
#java -d64 -Xmx10g -cp "marmot.jar;trove.jar" marmot.morph.cmd.Annotator -model-file models/btb/${lang}.marmot -lemmatizer-file models/btb/${lang}.lemming -test-file form-index=0,data/MorphoData-NewSplit/${lang}-${set}.txt.conv -pred-file predictions/btb/${lang}-${set}-pred.txt
