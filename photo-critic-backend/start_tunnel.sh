#!/bin/bash

# ssh -R 6666:localhost:6666 msucheck@mion.elka.pw.edu.pl -N
autossh -M 6667 -N -R 6666:localhost:6666 msucheck@mion.elka.pw.edu.pl &>> tunnel.log


