import sys
import logging
import argparse
import torch

from transformers import (
    pipeline, 
    BertJapaneseTokenizer, 
    AutoModelForQuestionAnswering
)


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

ROOT = '/home/miyawaki_shumpei/_QASRL-ja/work/QA_system/'


def create_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--context', type=str, help='')
    parser.add_argument('--question', type=str, help='')
    return parser


class QAModel(object):
    def __init__(self,
                 dir_model: str = ROOT+'outputs/',
                ):
        
        self.model = AutoModelForQuestionAnswering.from_pretrained(dir_model)  
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 

    def encode(self, question:str, context:str):
        return self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    def inference(self, inputs):
        return self.model(**inputs)

    def detect_span(self, output):
        start = torch.argmax(output.start_logits)  
        end = torch.argmax(output.end_logits) + 1 
        return start, end

    def decode_answer(self, input_ids, start, end):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[start:end]))

    def __call__(self, context, question):
        assert (question is not None) or (context is not None)
        inputs = self.encode(question, context)
        logger.info("In : " + self.tokenizer.decode(inputs['input_ids'][0]))
        input_ids = inputs["input_ids"].tolist()[0]
        output = self.inference(inputs)
        print(output)
        start, end = self.detect_span(output)
        answer = self.decode_answer(input_ids, start, end)
        logger.info("Out: " + answer)
        return answer


if __name__ == '__main__':
    
    qa_system = QAModel()

    #context = "私が詰め将棋の本を買ってきました。駒と盤は持っていません。"
    #for question in ["誰が買うか？", "何を買うか？", "何を持っているか？"]:
    #    answer = qa_system(context, question)
    #
    #    print("="*60)
    #    print("# CONTEXT  ::: ", context)
    #    print("# QUESTION ::: ", question)
    #    print("# ANSWER   ::: ", answer)

    parser = create_arg_parser()
    args = parser.parse_args()
    answer = qa_system(args.context, args.question)
    print('\033[32m' + answer + '\033[0m')
