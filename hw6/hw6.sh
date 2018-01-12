#!/bin/bash
wget -P ./ https://www.dropbox.com/s/uoqvqnzy5cy5u7q/encoder?dl=1 -O encoder
python3 test.py $1 $2 $3
