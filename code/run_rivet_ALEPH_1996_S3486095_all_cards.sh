#!/bin/bash
directory="ALEPH_1996_S3486095_Cards"

for i in {25..999}; do
	./main42 ALEPH_1996_S3486095_Cards/ALEPH_1996_S3486095_card_$i.cmnd ALEPH_1996_S3486095_card_$i.fifo &
	rivet -o rivet_histograms/simulation/ALEPH_1996_S3486095_card_$i.yoda -a ALEPH_1996_S3486095 ALEPH_1996_S3486095_card_$i.fifo

	rm ALEPH_1996_S3486095_card_$i.fifo
done
