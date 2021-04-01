import os
import sys
import json
import logging
import argparse

from tqdm import tqdm

import onnx
import onnxruntime as ort

import torch

import transformers
from transformers import BertJapaneseTokenizer, AutoModelForQuestionAnswering
from transformers.convert_graph_to_onnx import *


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def create_arg_parser(parser):
    # parser.add_argument('--model', type=str, help='')
    return parser


class OnnxConverter():
    pass


def print_args(args):
    print('\033[33m')
    print(args.__dict__)
    print('\033[0m')


def run():
    parser = OnnxConverterArgumentParser()
    parser = create_arg_parser(parser)
    args = parser.parse_args()
    args.output = Path(args.output).absolute()

    print_args(args)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)  
    tokenizer = BertJapaneseTokenizer.from_pretrained(args.tokenizer)

    try:
        print("\n====== Converting model to ONNX ======")
        # Convert
        convert(
            args.framework,
            model,
            args.output,
            args.opset,
            tokenizer,
            args.use_external_format,
            args.pipeline,
        )

        if args.quantize:
            # Ensure requirements for quantization on onnxruntime is met
            check_onnxruntime_requirements(ORT_QUANTIZE_MINIMUM_VERSION)

            # onnxruntime optimizations doesn't provide the same level of performances on TensorFlow than PyTorch
            if args.framework == "tf":
                print(
                    "\t Using TensorFlow might not provide the same optimization level compared to PyTorch.\n"
                    "\t For TensorFlow users you can try optimizing the model directly through onnxruntime_tools.\n"
                    "\t For more information, please refer to the onnxruntime documentation:\n"
                    "\t\thttps://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers\n"
                )

            print("\n====== Optimizing ONNX model ======")

            # Quantization works best when using the optimized version of the model
            args.optimized_output = optimize(args.output)

            # Do the quantization on the right graph
            args.quantized_output = quantize(args.optimized_output)

        # And verify
        if args.check_loading:
            print("\n====== Check exported ONNX model(s) ======")
            verify(args.output)

            if hasattr(args, "optimized_output"):
                verify(args.optimized_output)

            if hasattr(args, "quantized_output"):
                verify(args.quantized_output)

    except Exception as e:
        print("\033[31m" + f"Error while converting the model: {e}" + "\033[0m")
        exit(1)



if __name__ == '__main__':
    """ run
    % bash scripts/convert_onnx.sh
    """
    run()

'''
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


def convert_to_onnx(net, output_name, single_person, input_size):
    input = torch.randn(1, 3, input_size[0], input_size[1])
    input_layer_names = ['data']
    output_layer_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                          'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_layer_names,
                      output_names=output_layer_names)

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

import onnxruntime as ort

ort_session = ort.InferenceSession('alexnet.onnx')

outputs = ort_session.run(None, {'actual_input_1': np.random.randn(10, 3, 224, 224).astype(np.float32)})

print(outputs[0])
'''
