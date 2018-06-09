# Author: Bogomil Gospodinov
#!/bin/bash
# usage: mlp2pc [path-on-mlp2] [local-pc-path] [include-path-on-mlp2]
rsync -avzP -e 'ssh -o "ProxyCommand ssh -A s1312650@student.ssh.inf.ed.ac.uk nc %h %p"' \
	--include=\'$3\' \
	--exclude='*' \
	s1312650@mlp2:$1 $2