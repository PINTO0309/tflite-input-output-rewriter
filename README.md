# tflite-input-output-rewriter
This tool displays tflite signatures and rewrites the input/output OP name to the name of the signature. There is no need to install TensorFlow or TFLite.

## Environment
- Ubuntu 20.04+
- flatbuffers-compiler

## Motivation
- When TFLite models are generated, TensorFlow automatically prefixes the input OP name with `serving_default_`, resulting in very difficult-to-read models.

  ![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/094b5290-7d28-463d-80a6-4a485cd818e8)

- Also, the output OP name is arbitrarily rewritten to the unintelligible `StatefulPartitionedCall:n`.

  ![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/73bda215-f4d2-4fb0-9205-bff4a2e1fb45)

## Execution
```bash
sudo apt-get update && sudo apt-get install -y flatbuffers-compiler

python main.py -i xxxx.tflite
```

## Execution Result
- Inputs

  ![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/234d05fa-3926-4a51-a4f8-1e1fd4811304)

- Outputs

  ![image](https://github.com/PINTO0309/tflite-input-output-rewriter/assets/33194443/5ccea34b-9e98-4869-b9e7-9b45fdf6e987)
