#!/bin/bash
#docker run -v $PWD:$PWD -w $PWD -p 8888:8888 -it pythia8/tutorials:hsf23
# then jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root &

#docker run -v $PWD:$PWD -w $PWD -p 8889:8889 -it hepstore/rivet-pythia
docker run -v $PWD:$PWD -w $PWD -p 8890:8890 -it pythia_sbi_tune
