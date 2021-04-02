# Sentiment Analysis

## データセット
- [主観感情と客観感情の強度推定のための日本語データセット](https://github.com/ids-cv/wrime)
<img src="https://i.gyazo.com/1b235e24712423a2a0152e117fded5fa.png" title="wrime">
- [Twitter日本語評判分析データセット](http://www.db.info.gifu-u.ac.jp/data/Data_5d832973308d57446583ed9f)


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