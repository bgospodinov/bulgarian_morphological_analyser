#!/bin/bash
# Author: Bogomil Gospodinov
if [ "$#" -ne 1 ]; then
	echo 'Usage: ./delete_job.sh [job-id]'
	exit 1
fi

find models/ -path \*/\*${jobid}\* -print

read -p "Are you sure you want to cancel job and delete? (Yy/n) " -n 1 -r
echo # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	[[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

[ -x "$(command -v scancel)" ] && scancel ${jobid}
rm -rfv logs/*.${jobid}.log
find models/ -path \*/\*${jobid}\* -exec rm -rfv "{}" \;