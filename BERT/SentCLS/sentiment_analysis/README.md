# Sentiment Analysis

## データセット
- [主観感情と客観感情の強度推定のための日本語データセット](https://github.com/ids-cv/wrime)
<img src="https://i.gyazo.com/1b235e24712423a2a0152e117fded5fa.png" title="wrime" width="50%">

- [Twitter日本語評判分析データセット](http://www.db.info.gifu-u.ac.jp/data/Data_5d832973308d57446583ed9f)


## 辞書
- [日本語評価極性辞書](http://www.cl.ecei.tohoku.ac.jp/index.php?Open%20Resources%2FJapanese%20Sentiment%20Polarity%20Dictionary)
  - [oseti](https://github.com/ikegami-yukino/oseti)


## 二値分類
* https://huggingface.co/daigo/bert-base-japanese-sentiment

```py
>>> text = "私は幸福ある"
>>> config = "daigo/bert-base-japanese-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(config)
>>> model = AutoModelForSequenceClassification.from_pretrained(config)
>>> output = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)(text))
>>> output
[{'label': 'ポジティブ', 'score': 0.98430425}]
```


# References
* [意味クラブで作成したもの](https://github.com/cl-tohoku/Club-IMI-taiwa-2019/tree/master/sentiment_analysis/Japanese-sentiment-analysis/bert)
* https://colab.research.google.com/gist/hnishi/c3e7fbf352864aaffcef20e3bb4d5f9a/20200808_try_sentiment_analysis.ipynb#scrollTo=wo1kuGAz5a3_