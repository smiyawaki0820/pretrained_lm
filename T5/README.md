# T5

- [arXiv](https://arxiv.org/abs/1910.10683)
- [Google AI Blog](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
- [日本語解説](https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part7.html)

<img src="https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/img/pic202002-001.png" width="50%">

## In Japanese
- https://huggingface.co/sonoisa/t5-base-japanese
- [ソースコード (GitHub)](https://github.com/sonoisa/t5-japanese)
- [作成者 解説 (Qiita)](https://qiita.com/sonoisa/items/a9af64ff641f0bbfed44)
- Google Colab
  - [ニュース記事のジャンル予想（文章分類）](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb)
  - [ニュース記事のタイトル生成（一種の文章要約）](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_title_generation.ipynb)
  - [ニュース記事本文生成（文章生成）](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_article_generation.ipynb)
  - [ニュース記事のタイトル生成（一種の文章要約）の推論のみ](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_title_generation_inference.ipynb)
  - [ニュース記事本文生成（文章生成）の推論のみ](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_article_generation_inference.ipynb)

```py
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")  # 222M
```

### データセット
- [Wikipedia](https://ja.wikipedia.org/) の日本語ダンプデータ (2020年7月6日時点のもの)
- [OSCAR](https://oscar-corpus.com/) の日本語コーパス
- [CC-100](http://data.statmt.org/cc-100/) の日本語コーパス


