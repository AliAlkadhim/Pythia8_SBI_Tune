#!/bin/bash
echo "starting"

echo "shell" $0
# rnd=$(($1 + 1))

rnd=$(($1 + 15000 ))
BASE_DIR=.
#TUNES
echo $(pwd)
echo $(ls)
# STEP 1: SOURCE ENVIRONMENT
source /cvmfs/cms.cern.ch/cmsset_default.sh


source /cvmfs/sft.cern.ch/lcg/releases/LCG_88b/MCGenerators/rivet/3.1.6/x86_64-centos7-gcc62-opt/rivetenv.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh


export PYTHIA8DATA=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/pythia8/306-9f264/x86_64-centos7-gcc11-opt/share/Pythia8/xmldoc/

echo "hello"

# STEP 2: Run pythia with i'th .cmnd file
./main42 $BASE_DIR/PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_${rnd}.cmnd ALEPH_1996_S3486095_card_${rnd}.fifo &

rivet -o ALEPH_1996_S3486095_hist_${rnd}.yoda -a ALEPH_1996_S3486095 ALEPH_1996_S3486095_card_${rnd}.fifo

rm ALEPH_1996_S3486095_card_${rnd}.fifo
cp ALEPH_1996_S3486095_hist_${rnd}.yoda ALEPH_YODAS/
echo $1,$rnd
cd $BASE_DIR
