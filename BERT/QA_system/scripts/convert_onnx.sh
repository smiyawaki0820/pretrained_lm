#!/usr/bin/bash
set -ev

python src/convert_onnx.py \
  --pipeline question-answering \
  --model /home/miyawaki_shumpei/_QASRL-ja/work/QA_system/outputs \
  --tokenizer cl-tohoku/bert-base-japanese-whole-word-masking \
  --framework pt \
  --quantize \
  onnx_outputs/qarc.onnx \
  | tee logs/convert_onnx.log

