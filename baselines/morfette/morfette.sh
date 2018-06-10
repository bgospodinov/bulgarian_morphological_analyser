#!/bin/bash
#https://github.com/gchrupala/morfette

lang=bg

#morfette train --iter-lemma=20 --iter-pos=20 data/UD_Bulgarian-BTB/${lang}-train.conllu.conv models/${lang}
morfette predict models/${lang} < data/UD_Bulgarian-BTB/${lang}-dev.conllu.conv > predictions/${lang}_dev