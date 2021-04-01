import os
import sys
import json
import logging
import argparse

from tqdm import tqdm

from inference import QAModel


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

def timer(tag):
    def _timer(func):
        def wrapper(*args, **kwargs):
            import re
            import datetime
            funcname = re.compile('<function (?P<funcname>.*?) at').search(str(func)).group('funcname')
            logger.info(f'|--> START: {funcname}:{tag}')
            start = datetime.datetime.now()
            result = func(*args, **kwargs)
            end = datetime.datetime.now()
            logger.info(f'| TIME: {end-start}')
            logger.info(f'|--> END: {funcname}:{tag}')
            return result
        return wrapper
    return _timer


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fi_predict', default='outputs/nbest_predictions_ja.json', type=str)
    parser.add_argument('--fi_dev', default='data/DDQA-1.0/RC-QA/DDQA-1.0_RC-QA_dev.json', type=str)
    return parser


def validate(qa_system, fi_gold, fo_jsonl):
    dict_gold = json.load(open(fi_gold))
    """ dict_gold
    {
        'version': 'v2.0',
        'data': [{
            'title': '運転ドメイン',
            'paragraphs': [
                {
                    'context': str,
                    'qas': [{
                        'id': '5546..._00',
                        'question': str,
                        'answers': [{
                            'text': str,
                            'answer_start': int,
                        }],
                        'is_impossible': bool
                },
                {
                    ...
                }
                ]
            }]
        }]
    }
    """

    # print(len(dict_gold['data']))   # 1
    # print(dict_gold['data'][0].keys())  # ['title', 'paragraphs']
    # print(len(dict_gold['data'][0]['paragraphs']))  # 1053
    
    with open(fo_jsonl, 'w') as fo:
        count_all, count_em = 0, 0
        for line in tqdm(dict_gold['data'][0]['paragraphs']):
            _qas = []
            context = line['context']
            for qas in line['qas']:
                question = qas['question']
                answers = [ans['text'] for ans in qas['answers']]

                prediction = qa_system(context, question)
                is_match = (''.join(prediction.split()) in answers)
        
                count_all += 1
                count_em += bool(is_match)

                _qas.append({
                    'id': qas['id'],
                    'question': question,
                    'gold': answers,
                    'pred': prediction,
                    'is_match': is_match,
                    'is_impossible': qas['is_impossible'],
                })

            output = {
                'context': context,
                'qas': _qas
            }
            fo.write(json.dumps(output, ensure_ascii=False) + '\n')

        fo.write(f'EM score ... {count_em/count_all} ({count_em}/{count_all})')
        print(f'EM score ... {count_em/count_all} ({count_em}/{count_all})')


def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    qa_system = QAModel()
    validate(qa_system, args.fi_dev, 'outputs/predictions_ja.jsonl')

if __name__ == '__main__':
    run()
