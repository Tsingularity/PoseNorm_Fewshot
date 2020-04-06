#!/bin/bash
echo "downloading CUB..."
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
rm CUB_200_2011.tgz
rm attributes.txt
echo "CUB downloaded"

echo "downloading NABird..."
wget https://www.dropbox.com/s/nf78cbxq6bxpcfc/nabirds.tar.gz
tar -xzf nabirds.tar.gz
rm nabirds.tar.gz
echo "NABird downloaded"

echo "downloading FGVC-Aircraft..."
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xzf fgvc-aircraft-2013b.tar.gz
rm fgvc-aircraft-2013b.tar.gz
echo "FGVC-Aircraft downloaded"

echo "downloading OID..."
wget http://www.robots.ox.ac.uk/~vgg/data/oid/archives/oid-aircraft-beta-1.tar.gz
tar -xzf oid-aircraft-beta-1.tar.gz
rm oid-aircraft-beta-1.tar.gz
echo "OID downloaded"