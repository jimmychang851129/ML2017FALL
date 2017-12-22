#!/bin/bash
wget -P ./ https://www.dropbox.com/s/yrmkpwtcqncnzyz/bestmodel.hdf5?dl=1 -O bestmodel.hdf5
python3 test.py $1 $2
