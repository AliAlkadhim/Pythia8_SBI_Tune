#!/bin/bash
./main42 ALEPH_1996_S3486095_Cards/ALEPH_1996_S3486095_card_0.cmnd ALEPH_1996_S3486095_card_0.fifo &
rivet -o rivet_histograms/simulation/ALEPH_1996_S3486095_card_0.yoda -a ALEPH_1996_S3486095 ALEPH_1996_S3486095_card_0.fifo

rm ALEPH_1996_S3486095_card_0.fifo
yoda2root rivet_histograms/simulation/ALEPH_1996_S3486095_card_0.yoda rivet_histograms/simulation/ALEPH_1996_S3486095_card_0.root
