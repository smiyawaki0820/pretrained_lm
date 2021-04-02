# Sentiment Analysis with BERT

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