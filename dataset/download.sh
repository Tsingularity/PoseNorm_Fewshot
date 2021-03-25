#!/bin/bash
echo "downloading CUB..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45" -O CUB_200_2011.tgz && rm -rf /tmp/cookies.txt
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
