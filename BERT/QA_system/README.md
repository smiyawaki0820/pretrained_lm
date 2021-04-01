# これは何？
* RC-QA データセットで質問応答モデルをファインチューニングする
* ファインチューニングしたモデルを，収集したあアノテーションに使用できないか検証

## データセット
* [運転ドメインQAデータセット](http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/Driving%20domain%20QA%20datasets/download_ddqa.cgi) を使用する
> 運転ドメインQAデータセットは、ウェブ上で公開されている運転ドメインのブログ記事を基に構築しており、__述語項構造QAデータセット（PAS-QAデータセット）__と__文章読解QAデータセット（RC-QAデータセット）__から構成されています。
> (1) __PAS-QAデータセット__は、ガ格、ヲ格及びニ格について省略されている項の先行詞を問う問題であり、ガ格は12,468問、ヲ格は3,151問、ニ格は1,069問作成しました。
> (2) また、__RC-QAデータセット__は文章の中から質問に対する答えを抽出する問題であり、20,007問作成しました。
> これらのQAデータセットの作成には、大規模かつ短期間でデータセットを作成可能なクラウドソーシングを利用しました。QAデータセットの形式はSQuAD 2.0と同じです。PAS-QAデータセットのガ格とRC-QAデータセットは、全ての問題について文章中に答えがありますが、PAS-QAデータセットのヲ格とニ格は、一部の問題について文章中に答えが無く解答できないものがあります。データセットの構築方法と文章中に答えが無い問題については、参考文献をご参照ください。

* [高橋ら+'19, ドメインを限定した機械読解モデルに基づく述語項構造解析](https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/B1-4.pdf)

### PAS-QAデータセット
```txt
文章 : 私は右車線に移動した。バックミラーを見た。
質問 : “見た”の主語は何か？
答え : 私
```
### RC-QAデータセット
* 外界は考慮しないため、解答が外界のアノテーションについては注意が必要
```txt
文章 : 私の車の前をバイクにまたがった警察官が走っていた。
質問 : 警察官は何に乗っていた？
答え : バイク
```

## モデル
* transformers の [AutoModelForQuestionAnswering](https://huggingface.co/transformers/model_doc/auto.html#tfautomodelforquestionanswering) を使用する

### Fine-tuning
```bash
% bash scripts/run_squad.sh
```

__モデルの評価__
```bash
% python src/validate.py
EM score ... 0.7360454115421002 (778/1057)
```

### 推論
```bash
% python src/inference.py --context "詰め将棋の本を買ってきました。" --question "何を買うか？"
```

```python
>>> from src.inference import QAModel
>>> qa_system = QAModel()
>>> context = "詰め将棋の本を買ってきました。駒と盤は持っていません。"
>>> question = "何を買うか？"
>>> qa_system(context, question)
'詰め将棋 の 本'
```

### Convert to ONNX
```
% bash scripts/convert_onnx.sh
% python src/inference_with_onnx.py --context "詰め将棋の本を買ってきました。" --question "何を買うか？"
```


# 参考
* [Huggingface Transformersによる日本語の質問応答の学習](https://note.com/npaka/n/na8721fdc3e24)
* [Exporting Transformers Model](https://huggingface.co/transformers/serialization.html)
* [Accelerate your NLP pipelines using Hugging Face Transformers and ONNX Runtime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
* [transformers.convert\_graph\_to\_onnx.py](https://github.com/huggingface/transformers/blob/9e147d31f67a03ea4f5b11a5c7c3b7f8d252bfb7/src/transformers/convert_graph_to_onnx.py#L225)a
