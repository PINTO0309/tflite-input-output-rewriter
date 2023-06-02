import io
import os
import json
import requests
import subprocess
from typing import List, Dict

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

TFLITE_FILE = 'lite-model_movinet_a0_stream_kinetics-600_classification_tflite_float16_2_org.tflite'
OUTPUT_PATH = '.'

FBS_FILE_NAME = f'schema.fbs'
URL = f'https://raw.githubusercontent.com/tensorflow/tensorflow/v2.13.0-rc1/tensorflow/lite/schema/{FBS_FILE_NAME}'
fbs = requests.get(URL).content.decode()
with open(FBS_FILE_NAME, 'w') as f:
    f.write(fbs)

try:
    # Check to see if flatc is installed
    result = subprocess.check_output(
        [
            'flatc', '--version'
        ],
        stderr=subprocess.PIPE
    ).decode('utf-8')

    result = subprocess.check_output(
        [
            'flatc', '-t',
            '--strict-json',
            '--defaults-json',
            '-o', f'{OUTPUT_PATH}',
            f'{OUTPUT_PATH}/{FBS_FILE_NAME}',
            '--',
            f'{OUTPUT_PATH}/{TFLITE_FILE}',
        ],
        stderr=subprocess.PIPE
    ).decode('utf-8')

    # Rewrite input OP name and output OP name
    input_json_file_name = f'{os.path.splitext(os.path.basename(TFLITE_FILE))[0]}.json'
    input_json_file_path = f'{OUTPUT_PATH}/{input_json_file_name}'
    output_json_file_name = f'{os.path.splitext(os.path.basename(TFLITE_FILE))[0]}_renamed.json'
    output_json_file_path = f'{OUTPUT_PATH}/{output_json_file_name}'
    flat_json = None

    with open(input_json_file_path, 'r') as f:
        flat_json = json.load(f)

    flat_subgraphs = flat_json['subgraphs'][0]
    flat_tensors: List[Dict] = flat_subgraphs['tensors']
    flat_input_nums: List[int] = flat_subgraphs['inputs']
    flat_output_nums: List[int] = flat_subgraphs['outputs']
    flat_input_infos = [flat_tensors[idx] for idx in flat_input_nums]
    flat_output_infos = [flat_tensors[idx] for idx in flat_output_nums]

    flat_signature_def: Dict = flat_json['signature_defs'][0]
    flat_signature_def_inputs: List[Dict] = flat_signature_def['inputs']
    flat_signature_def_inputs_names = [flat_signature_def_input['name'] for flat_signature_def_input in flat_signature_def_inputs]
    flat_signature_def_outputs: List[Dict] = flat_signature_def['outputs']
    flat_signature_def_outputs_names = [flat_signature_def_output['name'] for flat_signature_def_output in flat_signature_def_outputs]

    # If the signature of the input OP and the signature of the output OP overlap, rename the signature of the output OP.
    for flat_signature_def_outputs_name in flat_signature_def_outputs_names:
        if flat_signature_def_outputs_name in flat_signature_def_inputs_names:
            rename_target_output = [
                flat_signature_def_output \
                    for flat_signature_def_output in flat_signature_def_outputs \
                        if flat_signature_def_output['name'] == flat_signature_def_outputs_name
            ][0]
            rename_target_output['name'] = f'output_{flat_signature_def_outputs_name}'

    # Rewrite input op names
    print('')
    print(f'{Color.GREEN}INFO: Overwriting the input OP name to the contents of signature_defs is in progress...{Color.RESET}')
    for flat_signature_def_input in flat_signature_def_inputs:
        tensor_index: int = flat_signature_def_input.get('tensor_index', -1)
        input_flat_tensor = [flat_tensor for flat_tensor in flat_tensors if int(flat_tensor['buffer']) == tensor_index + 1]
        if input_flat_tensor:
            input_flat_tensor = input_flat_tensor[0]
            print(f'{Color.GREEN}INFO:{Color.RESET} {Color.BLUE}FROM:{Color.RESET} {input_flat_tensor["name"]} {Color.BLUE}TO:{Color.RESET} {flat_signature_def_input["name"]}')
            input_flat_tensor['name'] = flat_signature_def_input['name']

    # Rewrite output op names
    print('')
    print(f'{Color.GREEN}INFO: Overwriting the output OP name to the contents of signature_defs is in progress...{Color.RESET}')
    for flat_signature_def_output in flat_signature_def_outputs:
        tensor_index: int = flat_signature_def_output.get('tensor_index', -1)
        output_flat_tensor = [flat_tensor for flat_tensor in flat_tensors if int(flat_tensor['buffer']) == tensor_index + 1]
        if output_flat_tensor:
            output_flat_tensor = output_flat_tensor[0]
            print(f'{Color.GREEN}INFO:{Color.RESET} {Color.BLUE}FROM:{Color.RESET} {output_flat_tensor["name"]} {Color.BLUE}TO:{Color.RESET} {flat_signature_def_output["name"]}')
            output_flat_tensor['name'] = flat_signature_def_output['name']

    with open(output_json_file_path, 'w') as f:
        json.dump(flat_json, f)

    # JSON -> tflite
    result = subprocess.check_output(
        [
            'flatc',
            '-o', f'{OUTPUT_PATH}',
            '-b', f'{OUTPUT_PATH}/schema.fbs',
            f'{output_json_file_path}'
        ],
        stderr=subprocess.PIPE
    ).decode('utf-8')
    # Delete JSON
    # os.remove(f'{output_json_file_path}')



except Exception as ex:
    print(
        f'{Color.YELLOW}WARNING:{Color.RESET} '+
        'Install "flatc". ' +
        'debian/ubuntu: apt-get install -y flatbuffers-compiler ' +
        'Other than debian/ubuntu: https://github.com/google/flatbuffers/releases'
    )