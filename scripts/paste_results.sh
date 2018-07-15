#!/bin/bash
for model in models/*$1 ; do
	printf "%s, %s, %s, %s \n" "$(echo $model | sed -e 's/.*_\([0-9]\+\)u_.*/\1/')" "${model#*/}" "m3_1_300_300" "$(cat $model/*/data/dev_avg_score 2>/dev/null)"
	[[ $2 == '-count' ]] && echo "$(ls -1q $model/*/data/dev_score.* | wc -l)"
done
