#!/bin/bash
# Download gs://openpi-assets/checkpoints/pi05_libero via public HTTPS (no gsutil). Resumable (curl -C -).
set -u
DST=$PROJ_ROOT/pi05/ckpt
cd $DST
curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&fields=items(name,size)&maxResults=1000" \
 | python3 -c "import json,sys; [print(i[\"name\"], i[\"size\"]) for i in json.load(sys.stdin)[\"items\"]]" > listing.txt
while read name size; do
  out=$DST/${name#checkpoints/}
  mkdir -p $(dirname $out)
  if [ -f $out ] && [ $(stat -c %s $out) = $size ]; then continue; fi
  for try in 1 2 3 4 5; do
    curl -s -L -C - -o $out "https://storage.googleapis.com/openpi-assets/$name"
    [ $(stat -c %s $out) = $size ] && break
    echo retry $name; sleep 5
  done
  echo "$(date +%T) $name $(stat -c %s $out)/$size"
done < listing.txt
# tokenizer used by openpi at runtime
mkdir -p $HOME/.cache/openpi/big_vision
curl -s -L -o $HOME/.cache/openpi/big_vision/paligemma_tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
ls -la $HOME/.cache/openpi/big_vision
echo DONE
