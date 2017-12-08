wget -P ./  https://www.dropbox.com/s/zhg8uzvp207rr9n/bestmodel.hdf5?dl=1 -O bestmodel.hdf5
wget -P ./ https://www.dropbox.com/s/wgtkwvivbetuar5/vocab?dl=1 -O vocab
wget -P ./ https://www.dropbox.com/s/r5iwvygla7aizxp/vocab.syn1neg.npy?dl=1 -O vocab.syn1neg.npy
wget -P ./ https://www.dropbox.com/s/ab7tvpmwgzexs4b/vocab.wv.syn0.npy?dl=1 -O vocab.wv.syn0.npy
python3 upload.py $1 $2
