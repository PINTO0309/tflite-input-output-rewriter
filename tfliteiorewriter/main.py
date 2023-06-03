import os
import sys
import json
import requests
import subprocess
from typing import List, Dict, Optional
from argparse import ArgumentParser

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


def rewrite(
    *,
    tflite_file: str,
    view_mode: Optional[bool] = False,
    output_path: Optional[str] = '.',
    rename_list: Optional[List[List[str]]] = [],
):
    """Rewrite tflite input/output names.

    Parameters
    ----------
    tflite_file: str
        Input tflite file path.

    view_mode: Optional[bool]
        Runs in a mode that only displays the signature_defs recorded in the model.
        This mode does not rewrite the model.
        Default: False

    output_path: Optional[str]
        Output tflite file folder path.
        Default: "."

    rename_list: Optional[List[List[str]]]
        Replace with any specified name.
        rename_list = [[{from_name1}, {to_name1}], [{from_name2}, {to_name2}], [{from_name3}, {to_name3}]]
        Default: []
    """
    TF_VER: str = 'v2.13.0-rc1'
    FBS_FILE_NAME: str = f'schema.fbs'
    URL: str = f'https://raw.githubusercontent.com/tensorflow/tensorflow/{TF_VER}/tensorflow/lite/schema/{FBS_FILE_NAME}'

    # Download schema.fbs
    if not os.path.exists(FBS_FILE_NAME):
        fbs = requests.get(URL).content.decode()
        with open(FBS_FILE_NAME, 'w') as f:
            f.write(fbs)

    try:
        # Check to see if flatc is installed
        _ = subprocess.check_output(
            [
                'flatc', '--version'
            ],
            stderr=subprocess.PIPE
        ).decode('utf-8')

        _ = subprocess.check_output(
            [
                'flatc', '-t',
                '--strict-json',
                '--defaults-json',
                '-o', f'{output_path}',
                f'{output_path}/{FBS_FILE_NAME}',
                '--',
                f'{output_path}/{tflite_file}',
            ],
            stderr=subprocess.PIPE
        ).decode('utf-8')

        # Rewrite input OP name and output OP name
        input_json_file_name = f'{os.path.splitext(os.path.basename(tflite_file))[0]}.json'
        input_json_file_path = f'{output_path}/{input_json_file_name}'
        output_json_file_name = f'{os.path.splitext(os.path.basename(tflite_file))[0]}_renamed.json'
        output_json_file_path = f'{output_path}/{output_json_file_name}'
        flat_json = None

        with open(input_json_file_path, 'r') as f:
            flat_json = json.load(f)

        # Checks if signature_defs are recorded in tflite
        if 'signature_defs' not in flat_json or not flat_json['signature_defs']:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} ' +
                f'Processing is aborted because signature_defs is not recorded in tflite.'
            )
            sys.exit(0)

        flat_subgraphs = flat_json['subgraphs'][0]
        flat_tensors: List[Dict] = flat_subgraphs['tensors']
        flat_signature_def: Dict = flat_json['signature_defs'][0]
        flat_signature_def_inputs: List[Dict] = flat_signature_def['inputs']
        flat_signature_def_inputs_names = [
            flat_signature_def_input['name'] \
                for flat_signature_def_input in flat_signature_def_inputs
        ]
        flat_signature_def_outputs: List[Dict] = flat_signature_def['outputs']
        flat_signature_def_outputs_names = [
            flat_signature_def_output['name'] \
                for flat_signature_def_output in flat_signature_def_outputs
        ]

        # If the signature of the input OP and the signature of the output OP overlap,
        # rename the signature of the output OP.
        if not view_mode:
            if not rename_list:
                # Override OP name by name in signature_defs
                for flat_signature_def_outputs_name in flat_signature_def_outputs_names:
                    if flat_signature_def_outputs_name in flat_signature_def_inputs_names:
                        rename_target_output = [
                            flat_signature_def_output \
                                for flat_signature_def_output in flat_signature_def_outputs \
                                    if flat_signature_def_output['name'] == flat_signature_def_outputs_name
                        ][0]
                        rename_target_output['name'] = f'output_{flat_signature_def_outputs_name}'
            else:
                # Override OP name by name in rename_list
                for flat_signature_def_input in flat_signature_def_inputs:
                    tensor_index: int = flat_signature_def_input.get('tensor_index', -1)
                    input_flat_tensor = [
                        flat_tensor \
                            for flat_tensor in flat_tensors \
                                if int(flat_tensor['buffer']) == tensor_index + 1
                    ]
                    if input_flat_tensor:
                        input_flat_tensor = input_flat_tensor[0]
                        for rename_set in rename_list:
                            # rename_set[0]: From, rename_set[1]: To
                            if input_flat_tensor['name'] == rename_set[0]:
                                flat_signature_def_input['name'] = rename_set[1]

                for flat_signature_def_output in flat_signature_def_outputs:
                    tensor_index: int = flat_signature_def_output.get('tensor_index', -1)
                    output_flat_tensor = [
                        flat_tensor \
                            for flat_tensor in flat_tensors \
                                if int(flat_tensor['buffer']) == tensor_index + 1
                    ]
                    if output_flat_tensor:
                        output_flat_tensor = output_flat_tensor[0]
                        for rename_set in rename_list:
                            # rename_set[0]: From, rename_set[1]: To
                            if output_flat_tensor['name'] == rename_set[0]:
                                flat_signature_def_output['name'] = rename_set[1]

            # Rewrite input op names
            print('')
            print(f'{Color.GREEN}INFO: Overwriting the input OP name to the contents of signature_defs is in progress...{Color.RESET}')
            for flat_signature_def_input in flat_signature_def_inputs:
                tensor_index: int = flat_signature_def_input.get('tensor_index', -1)
                input_flat_tensor = [
                    flat_tensor \
                        for flat_tensor in flat_tensors \
                            if int(flat_tensor['buffer']) == tensor_index + 1
                ]
                if input_flat_tensor:
                    input_flat_tensor = input_flat_tensor[0]
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} ' +
                        f'{Color.BLUE}FROM:{Color.RESET} {input_flat_tensor["name"]} ' +
                        f'{Color.BLUE}TO:{Color.RESET} {flat_signature_def_input["name"]}'
                    )
                    input_flat_tensor['name'] = flat_signature_def_input['name']

            # Rewrite output op names
            print('')
            print(f'{Color.GREEN}INFO: Overwriting the output OP name to the contents of signature_defs is in progress...{Color.RESET}')
            for flat_signature_def_output in flat_signature_def_outputs:
                tensor_index: int = flat_signature_def_output.get('tensor_index', -1)
                output_flat_tensor = [
                    flat_tensor \
                        for flat_tensor in flat_tensors \
                            if int(flat_tensor['buffer']) == tensor_index + 1
                ]
                if output_flat_tensor:
                    output_flat_tensor = output_flat_tensor[0]
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} ' +
                        f'{Color.BLUE}FROM:{Color.RESET} {output_flat_tensor["name"]} ' +
                        f'{Color.BLUE}TO:{Color.RESET} {flat_signature_def_output["name"]}'
                    )
                    output_flat_tensor['name'] = flat_signature_def_output['name']

        else:
            # Print signature_defs
            # Inputs
            print('')
            print(f'{Color.GREEN}INFO: Input signature_defs...{Color.RESET}')
            for flat_signature_def_input in flat_signature_def_inputs:
                tensor_index: int = flat_signature_def_input.get('tensor_index', -1)
                input_flat_tensor = [
                    flat_tensor \
                        for flat_tensor in flat_tensors \
                            if int(flat_tensor['buffer']) == tensor_index + 1
                ]
                if input_flat_tensor:
                    input_flat_tensor = input_flat_tensor[0]
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} ' +
                        f'{Color.BLUE}NAME:{Color.RESET} {flat_signature_def_input["name"]} ' +
                        f'{Color.BLUE}TYPE:{Color.RESET} {input_flat_tensor["type"]} ' +
                        f'{Color.BLUE}SHAPE:{Color.RESET} {input_flat_tensor["shape"]} ' +
                        f'{Color.MAGENTA}OPNAME:{Color.RESET} {input_flat_tensor["name"]}'
                    )

            # Outputs
            print('')
            print(f'{Color.GREEN}INFO: Output signature_defs...{Color.RESET}')
            for flat_signature_def_output in flat_signature_def_outputs:
                tensor_index: int = flat_signature_def_output.get('tensor_index', -1)
                output_flat_tensor = [
                    flat_tensor \
                        for flat_tensor in flat_tensors \
                            if int(flat_tensor['buffer']) == tensor_index + 1
                ]
                if output_flat_tensor:
                    output_flat_tensor = output_flat_tensor[0]
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} ' +
                        f'{Color.BLUE}NAME:{Color.RESET} {flat_signature_def_output["name"]} ' +
                        f'{Color.BLUE}TYPE:{Color.RESET} {output_flat_tensor["type"]} ' +
                        f'{Color.BLUE}SHAPE:{Color.RESET} {output_flat_tensor["shape"]} ' +
                        f'{Color.MAGENTA}OPNAME:{Color.RESET} {output_flat_tensor["name"]}'
                    )
            sys.exit(0)

        with open(output_json_file_path, 'w') as f:
            json.dump(flat_json, f)

        # JSON -> tflite
        _ = subprocess.check_output(
            [
                'flatc',
                '-o', f'{output_path}',
                '-b', f'{output_path}/schema.fbs',
                f'{output_json_file_path}'
            ],
            stderr=subprocess.PIPE
        ).decode('utf-8')

        # Delete JSON
        # os.remove(f'{input_json_file_path}')
        # os.remove(f'{output_json_file_path}')

    except Exception as ex:
        print(
            f'{Color.YELLOW}WARNING:{Color.RESET} '+
            'Install "flatc". ' +
            'debian/ubuntu: apt-get install -y flatbuffers-compiler ' +
            'Other than debian/ubuntu: https://github.com/google/flatbuffers/releases'
        )

def cli():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_tflite_file_path',
        required=True,
        type=str,
        help='Input tflite file path.'
    )
    parser.add_argument(
        '-v',
        '--view',
        action='store_true',
        help=\
            'Runs in a mode that only displays the signature_defs recorded in the model. ' +
            'This mode does not rewrite the model.'
    )
    parser.add_argument(
        '-o',
        '--output_folder_path',
        type=str,
        default='.',
        help='Output tflite file folder path.'
    )
    parser.add_argument(
        '-r',
        '--rename',
        type=str,
        nargs=2,
        action='append',
        help=\
            'Replace with any specified name. ' +
            '--rename {from_name1} {to_name1} --rename {from_name2} {to_name2} --rename {from_name3} {to_name3}'
    )
    args = parser.parse_args()
    tflite_file: str = args.input_tflite_file_path
    view_mode: bool = args.view
    output_path: str = args.output_folder_path
    rename_list: List[List[str]] = args.rename
    rewrite(
        tflite_file=tflite_file,
        view_mode=view_mode,
        output_path=output_path,
        rename_list=rename_list,
    )

if __name__ == '__main__':
    cli()

