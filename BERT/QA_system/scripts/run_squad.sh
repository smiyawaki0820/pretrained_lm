ROOT=/home/miyawaki_shumpei/_QASRL-ja/work/QA_system
DATA=$ROOT/data/DDQA-1.0/RC-QA/DDQA-1.0_RC-QA
DEST=$ROOT/outputs

mkdir -p $DEST

cd $ROOT/transformers

python examples/legacy/question-answering/run_squad.py \
  --model_type=bert \
  --model_name_or_path=cl-tohoku/bert-base-japanese-whole-word-masking \
  --tokenizer_name=cl-tohoku/bert-base-japanese-whole-word-masking \
  --do_train \
  --do_eval \
  --train_file=${DATA}_train.json \
  --predict_file=${DATA}_dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir ${DEST} \
| tee $ROOT/logs/run_squad.log
