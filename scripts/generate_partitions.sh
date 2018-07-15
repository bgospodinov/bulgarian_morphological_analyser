#!/bin/bash

for f in train dev test ; do 
	awk '{ print $2, $3, $4 }' <(cat $f/*) > ${f}.txt
	dos2unix ${f}.txt
done

mv train.txt training.txt