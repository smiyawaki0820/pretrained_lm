import os
import sys

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


class Processor():
    def __init__(self, ):
        tokenizer = AutoTokenizer.from_pretrained("daigo/bert-base-japanese-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("daigo/bert-base-japanese-sentiment")