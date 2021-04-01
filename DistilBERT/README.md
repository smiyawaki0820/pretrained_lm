# Distil BERT
* [Sanh+'19: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (NeurIPS)](https://arxiv.org/abs/1910.01108)

> BERT-base よりもパラメータが 40% 少なく、60% 高速に動作し、GLUE Benchmark で測定された BERT の 97% の性能を維持できる知識蒸留の枠組みを使用したモデル。


## In English
* https://huggingface.co/transformers/model_doc/distilbert.html

## In Japanese
* https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp
* https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp/blob/main/docs/GUIDE.md

* [公式ガイドライン](https://github.com/huggingface/transformers/tree/master/examples/distillation)に従い、`6-layer, 768-hidden, 12-heads, 66M parameters`のモデルを学習。

```bash
$ python transformers/examples/distillation/train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-japanese-whole-word-masking \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path $MODEL_PATH \
    --data_file $BINARY_DATA_PATH \
    --token_counts $COUNTS_TOKEN_PATH
```

### Usage

```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")
model = AutoModel.from_pretrained("bandainamco-mirai/distilbert-base-japanese")  
```

### ライブドアニュースコーパスを使用したテキスト分類（ファインチューニング）
* https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp/blob/main/docs/GUIDE.md#%E8%87%AA%E5%89%8D%E3%81%AE%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%81%A7fine-tuning