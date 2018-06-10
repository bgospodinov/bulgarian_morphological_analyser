#Author: Toms Bergmanis toms.bergmanis@gmail.com

#The get the latest versions of UD (v2.1):
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2515{/ud-treebanks-v2.1.tgz}
tar xvzf ud-treebanks-v2.1.tgz
rm ud-treebanks-v2.1.tgz


#To get UD-v2.0 version that was used in the original Lematus paper for all languages except Dutch use:
#curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1983{/ud-treebanks-v2.0.tgz}
#tar xvzf ud-treebanks-v2.0.tgz
#rm ud-treebanks-v2.0.tgz
# and then use this to get the test sets
#TODO currently, for version v2.0, the test sets have to be added and renamed to match their coresponding train and dev sets manually: write a script for it.
#curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2184{/ud-test-v2.0-conll2017.tgz}
#tar xvzf ud-test-v2.0-conll2017.tgz
#rm ud-test-v2.0-conll2017.tgz

