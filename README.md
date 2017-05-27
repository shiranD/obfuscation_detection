# Obfuscation Detection
The primary goal of this project is to detect the obfuscation method used to encode the malware script.
This project is a proof of concept to describe an encoding language detector for different obfuscated parts of the code. One part learns to recognize 3 different obfuscation methods. The second pard learns to recognize the XOR key, given that the first part predicted the method to be XOR. In the third part there is a handy tool that finds the obfuscated locations and provide a prediction for their type. 
