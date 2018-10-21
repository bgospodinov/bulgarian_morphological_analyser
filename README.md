# msc_project
University of Edinburgh MSc project 2017/2018\
Bogomil Gospodinov (s1312650@sms.ed.ac.uk)

A contextual morphological analyzer for Bulgarian, based on the attentive encoder-decoder machine translation system Nematus. Read more here: https://github.com/bgospodinov/msc_report

## How to install
1. Install conda dependencies `conda env create -f=msc.yml -n [envname]`
1. Run `git submodule update --init` to initialise submodule dependencies

## Preprocessing Universal Dependencies datasets for lemmatization
In order to be lemmatized by Nematus, each dataset partition has to first be transformed into parallel "source" and "target" language files, corresponding to the source and target language in a machine translation task.
 1. We start by transforming the training, dev and test sets into target and source files using the following script\
	`python -m data.transform_ud --input [path-to-partition-file]`\
	By default transform_ud.py outputs two files per partition (training, set or dev) inside the **input/** folder (relative to the script path). The files can be found in a directory with a name containing the name of the dataset and a subset of the script's arguments e.g. **input/MorphoData-NewSplit_20_char_1**. The name of the files will contain the name of the original input file with "_source" or "_target" appended to it. You can specify custom paths using --output [source_path] [target_path]. To overwrite existing files in an existing directory pass --overwrite.\
There are many possible options to modify the output of transform_ud.py. Among the most important are: --context_unit (word, char, bpe), --word_unit (word, char), --tag_unit (word, char), --context_char_n_gram (1 if context_unit is char, ignored otherwise) (which determine the level of representation of the respective entity); --context_size or --context_char_size (which determine the size of the left- and right-hand side context for each word); --context_span (which determines how many sentences to include in the potential context of the word on both left and right).\
Type `python -m data.transform_ud -h` to list all options.
 1. Next we need to build a dictionary out of the vocabulary used in the new transformed partitions. We can do that by using Nematus's inbuilt scripts. Perform this only for the training source and target.\
 `./nematus/data/build_dictionary.py [path_to_training_source] [path_to_training_target]`\
 Two .json dictionary files will be generated for the target and source vocabularies found in the training set. These will be placed in the same directory as the training files.
 
## Training
Look at `scripts/sample_train.sh` or `scripts/sample_batch_train.sh` for example training sessions. Execute `scripts/sample_translate.sh` after sample_train.sh for sample translation. All scripts must be executed from the root dir of the project.
