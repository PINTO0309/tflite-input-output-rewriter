# tflite-input-output-rewriter
This tool displays tflite signatures and rewrites the input/output OP name to the name of the signature. There is no need to install TensorFlow or TFLite.

## Environment
- Ubuntu 20.04+
- flatbuffers-compiler

## Motivation
The purpose is to solve the following problems by forcibly rewriting tflite's input/output OP names.

- When TFLite models are generated, TensorFlow automatically prefixes the input OP name with `serving_default_`, resulting in very difficult-to-read models. Also, an unnecessary index `:n` is added to the end of the name.

  ![01](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/c83f4722-aca6-4fd6-910e-b23b20357706)

- Also, the output OP name is arbitrarily rewritten to the unintelligible `StatefulPartitionedCall:n`.

  ![02](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/5d73d9e1-cae3-498f-8de6-371a8ddb9ce6)

## Execution
```bash
sudo apt-get update && sudo apt-get install -y flatbuffers-compiler

python main.py -i xxxx.tflite
```
```
usage: main.py [-h] -i INPUT_TFLITE_FILE_PATH [-v] [-o OUTPUT_FOLDER_PATH]

optional arguments:
  -h, --help
      show this help message and exit

  -i INPUT_TFLITE_FILE_PATH, --input_tflite_file_path INPUT_TFLITE_FILE_PATH
      Input tflite file path.

  -v, --view
      Runs in a mode that only displays the signature_defs recorded in the model.
      This mode does not rewrite the model.

  -o OUTPUT_FOLDER_PATH, --output_folder_path OUTPUT_FOLDER_PATH
      Output tflite file folder path.
```

![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/d676da7d-533f-4fca-b5c5-09a737ffb118)

![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/d58dca0f-ac51-4545-b49b-32f22e7a39ad)

## Execution Result
- Inputs

  ![03](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/b0b4bf83-bbcf-4a26-aaf9-86e9feaf69de)

- Outputs

  ![04](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/bedffe08-c072-4b07-af8f-d763a2708907)

## View Mode Result
```bash
python main.py -i xxxx.tflite -v
```
