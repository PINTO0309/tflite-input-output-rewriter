# tflite-input-output-rewriter
This tool displays tflite signatures and rewrites the input/output OP name to the name of the signature. There is no need to install TensorFlow or TFLite.

[![Downloads](https://static.pepy.tech/personalized-badge/tfliteiorewriter?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/tfliteiorewriter) ![GitHub](https://img.shields.io/github/license/PINTO0309/tflite-input-output-rewriter?color=2BAF2B) [![Python](https://img.shields.io/badge/Python-3.8-2BAF2B)](https://img.shields.io/badge/Python-3.8-2BAF2B) [![PyPI](https://img.shields.io/pypi/v/tfliteiorewriter?color=2BAF2B)](https://pypi.org/project/tfliteiorewriter/)

## Environment
- Ubuntu 20.04+
- flatbuffers-compiler
- requests

## Motivation
The purpose is to solve the following problems by forcibly rewriting tflite's input/output OP names.

- When TFLite models are generated, TensorFlow automatically prefixes the input OP name with `serving_default_`, resulting in very difficult-to-read models. Also, an unnecessary index `:n` is added to the end of the name.

  ![01](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/c83f4722-aca6-4fd6-910e-b23b20357706)

- Also, the output OP name is arbitrarily rewritten to the unintelligible `StatefulPartitionedCall:n`.

  ![02](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/5d73d9e1-cae3-498f-8de6-371a8ddb9ce6)

## Execution
1. Docker
    ```
    $ docker login ghcr.io

    Username (xxxx): {Enter}
    Password: {Personal Access Token}
    Login Succeeded

    $ docker run --rm -it \
    -v `pwd`:/home/user \
    ghcr.io/pinto0309/tflite-input-output-rewriter:latest

    $ tfliteiorewriter -i xxxx.tflite
    ```
2. Local
    ```bash
    $ sudo apt-get update && sudo apt-get install -y flatbuffers-compiler
    # Other than debian/ubuntu: https://github.com/google/flatbuffers/releases
    $ pip install -U tfliteiorewriter

    $ tfliteiorewriter -i xxxx.tflite
    ```
```
usage: tfliteiorewriter
  [-h]
  -i INPUT_TFLITE_FILE_PATH
  [-v]
  [-o OUTPUT_FOLDER_PATH]
  [-r RENAME RENAME]

optional arguments:
  -h, --help
      show this help message and exit

  -i INPUT_TFLITE_FILE_PATH, --input_tflite_file_path INPUT_TFLITE_FILE_PATH
      Input tflite file path.
      If `--rename` is not used, the input/output OP name is overwritten with the definition
      information in signature_defs.

  -v, --view
      Runs in a mode that only displays the signature_defs recorded in the model.
      This mode does not rewrite the model.

  -o OUTPUT_FOLDER_PATH, --output_folder_path OUTPUT_FOLDER_PATH
      Output tflite file folder path.

  -r RENAME RENAME, --rename RENAME RENAME
      Replace with any specified name.
      --rename {from_name1} {to_name1} --rename {from_name2} {to_name2} ...
      --rename serving_default_input_1:0 aaa --rename StatefulPartitionedCall:0 bbb
```

![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/d676da7d-533f-4fca-b5c5-09a737ffb118)

![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/d58dca0f-ac51-4545-b49b-32f22e7a39ad)

## Execution Result
If this tool is run without additional options, it will overwrite the input/output OP names for Netron display with the input/output name definition information in `signature_defs`.
- Inputs

  ![03](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/b0b4bf83-bbcf-4a26-aaf9-86e9feaf69de)

- Outputs

  ![04](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/bedffe08-c072-4b07-af8f-d763a2708907)

## View Mode Result
```bash
tfliteiorewriter -i xxxx.tflite -v
```
![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/0d43d93d-647d-40f1-b464-662e39dcf228)

## Rename Mode Result
Replace with any name by specifying `{From}` and `{To}` in the `--renmae (-r)` option.

- Before

  ![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/61195485-a756-4449-8bf2-4d9e83f06feb)

```bash
tfliteiorewriter \
-i xxxx.tflite \
-r serving_default_input_1:0 aaa \
-r StatefulPartitionedCall:0 bbb
```
![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/e2b33158-e044-460f-b032-747339e86feb)

- After

  ![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/b339856a-63aa-46bf-9e3c-4f65198b346a)

