# Obfuscation Encoding Detection
The run.sh script provided here train three different models to detect different obfuscation encodings, and the script can be modified with more encodings. The model that is created contains plaintext class as well as some classes tend to be confused with it. The model is a simple LSTM that learns from a snippent of 100 byte sequence of obfuscated code. Evaluation is employed at the end of training.

The input for the script is potential code to obfuscate. Real world scripts of malware are ideal. Throughout the shell script the will be new folders generated with intermediate output that are essential for the scripts. Ensure premissions are provided.
