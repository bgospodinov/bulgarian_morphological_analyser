#Partial credit to Tom Bergmannis
#read about these in lemmings web page
options=use-morph=false,use-perceptron=false
options=$options,use-mallet=false,offline-feature-extraction=false
options=$options,tag-dependent=true,seed=12345

#language extension
lang=bg
set=dev
data=UD_Bulgarian-BTB

mkdir -p models

#format of the train file as below
#можем \s+ мога \s+ Vpiif-r1p
# empty line used as a sentence separator

#train POS model
#mkdir -p models/${data} && java -d64 -Xmx10G -cp marmot.jar marmot.morph.cmd.Trainer -tag-morph false -train-file form-index=0,tag-index=2,../../data/datasets/${data}/training.txt -model-file models/${data}/${lang}.marmot

#train lemmatizer
#java -d64 -Xmx10G -cp "marmot.jar;mallet.jar;trove.jar" lemming.lemma.cmd.Trainer lemming.lemma.ranker.RankerTrainer $options models/${data}/${lang}.lemming form-index=0,lemma-index=1,tag-index=2,../../data/datasets/${data}/training.txt

#make predictions 
#mkdir -p predictions/${data} && java -d64 -Xmx10g -cp "marmot.jar;trove.jar" marmot.morph.cmd.Annotator -model-file models/${data}/${lang}.marmot -lemmatizer-file models/${data}/${lang}.lemming -test-file form-index=0,../../data/datasets/${data}/${set}.txt -pred-file predictions/${data}/${lang}-${set}-pred.txt
#./predictions/ud/convert_lemming_pred.awk predictions/${data}/${lang}-${set}-pred.txt > predictions/${data}/${lang}-${set}-pred-py.txt
