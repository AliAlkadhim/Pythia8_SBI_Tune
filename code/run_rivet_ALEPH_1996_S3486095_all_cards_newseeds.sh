#!/bin/bash
directory="ALEPH_1996_S3486095_CaALEPH_1996_S3486095_Cards_newseedsrds"

for i in {0..999}; do
	./main42 ALEPH_1996_S3486095_Cards_newseeds/ALEPH_1996_S3486095_card_newseed_$i.cmnd ALEPH_1996_S3486095_card_$i.fifo &
	rivet -o rivet_histograms/simulation/newseeds/ALEPH_1996_S3486095_card_newseed_$i.yoda -a ALEPH_1996_S3486095 ALEPH_1996_S3486095_card_$i.fifo

	rm ALEPH_1996_S3486095_card_$i.fifo
done
