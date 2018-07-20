#!/bin/bash

for f in train dev ; do 
	cat $f/* | expand > ${f}.txt
	dos2unix ${f}.txt
	awk 'NF > 0 {blank=0} NF == 0 {blank++} blank < 2 && NR>1 { print $2, $3, $4 }' ${f}.txt > ${f}.tmp.txt
	rm -f ${f}.txt
	mv ${f}.tmp.txt ${f}.txt
	sed -i.bak -r -e 's/^\s+$//g' ${f}.txt
done

mv train.txt training.txt
rm -f *.bak