# XOR Key Detection
The run.sh script provided here train three different models to detect the XOR key a file was ciphered with. The keys are between 1-256 (code can be modified to learn more than that though). The models are a simple 100 byte LSTM, ensemble of independent LSTMs to learn each bit in the key given the input, and a recurrent ensemble of LSTMs that is provided with the input sequence of ciphered text and the previous prediction for it. Evaluation is provided as well. Evaluation has many options its specifics can be modified.

The input for the script is potential code to obfuscate. Real world scripts of malware are ideal. Throughout the shell script the will be new folders generated with intermediate output that are essential for the scripts. Ensure premissions are provided.
