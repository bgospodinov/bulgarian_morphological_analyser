#!/bin/bash
# $1 - predictions file
# $2 - gold standard file
# NB: DONT FORGET TO CONVERT PREDICTION FILE TO UTF-8 w/o BOM
# example  . predictions/compute_pred_ud.sh predictions/ud/bg-dev-pred.txt data/UD_Bulgarian-BTB/bg-dev.txt.conv
# example  . predictions/compute_pred_ud.sh predictions/ud/bg-test-pred.txt data/UD_Bulgarian-BTB/bg-test.txt.conv

# total number of lines, we assume both files have the same
total=$(wc -l $2 | cut -d' ' -f1)

# lemmatization accuracy
lemma_wrong=`diff --strip-trailing-cr -U 0 <(cut -d$'\t' -f4 $1) <(cat $2 | tr -s ' ' | cut -d' ' -f2) | grep -c ^@`
lemma_acc=`python -c "print(1 - ($lemma_wrong/($total*1.0)))"`
echo "Lemma accuracy: $lemma_acc"

# pos accuracy
pos_wrong=`diff --strip-trailing-cr -U 0 <(cut -d$'\t' -f6 $1) <(cat $2 | tr -s ' ' | cut -d' ' -f3) | grep -c ^@`
pos_acc=`python -c "print(1 - ($pos_wrong/($total*1.0)))"`
echo "POS accuracy: $pos_acc"

# morph accuracy
morph_wrong=`diff --strip-trailing-cr -U 0 <(cut -d$'\t' -f8 $1) <(cat $2 | tr -s ' ' | cut -d' ' -f4) | grep -c ^@`
morph_acc=`python -c "print(1 - ($morph_wrong/($total*1.0)))"`
echo "Morph accuracy: $morph_acc"

# joint accuracy
joint_wrong=`diff --strip-trailing-cr -U 0 <(cut -d$'\t' -f4,6 $1) <(tr ' ' $'\t' < <(cat $2 | tr -s ' ' | cut -d' ' -f2,3)) | grep -c ^@`
joint_acc=`python -c "print(1 - ($joint_wrong/($total*1.0)))"`
echo "Joint accuracy: $joint_acc"