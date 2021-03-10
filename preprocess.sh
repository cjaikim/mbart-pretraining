DICT=dict.txt
SRC=de_DE
TGT=hsb
DATA='./dataset'
TRAIN='train'
TEST='test'
DEST='preprocessed'
NAME='processed_set'

fairseq-preprocess \
  --task multilingual_denoising \
  --trainpref ${DATA}/${TRAIN}.spm \
  --testpref ${DATA}/${TEST}.spm \
  --destdir ${DEST}/${NAME} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
