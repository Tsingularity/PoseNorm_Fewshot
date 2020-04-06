#!/bin/bash
echo "initializing CUB..."
python init_cub.py --origin_path ./CUB_200_2011
echo "CUB finished"

echo "initializing NABird..."
python init_na.py --origin_path ./nabirds
echo "NABird finished"

echo "initializing FGVC-Aircraft..."
python init_fgvc.py --origin_path ./fgvc-aircraft-2013b
echo "FGVC-Aircraft finished"

echo "initializing OID..."
python init_oid.py \
    --oid_origin_path ./oid-aircraft-beta-1 \
    --fgvc_origin_path ./fgvc-aircraft-2013b
echo "OID finished"