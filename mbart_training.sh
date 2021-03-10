#!/bin/bash
langs=de_DE,hsb

fairseq-train preprocessed/processed_set \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_base_wmt20 --layernorm-embedding \
  --task multilingual_denoising \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr 3e-05 --warmup-updates 2500 \
  --dropout 0.3 --attention-dropout 0.1 \
  --weight-decay 0.0 --max-tokens 512 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 \
  --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 \
  --langs $langs
  
