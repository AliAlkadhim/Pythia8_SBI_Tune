#!/bin/bash
./main42 test_card.cmnd ALEPH_1996_S3486095_card_0.fifo &
rivet -o test_card.yoda -a ALEPH_1996_S3486095 ALEPH_1996_S3486095_card_0.fifo

rm ALEPH_1996_S3486095_card_0.fifo
