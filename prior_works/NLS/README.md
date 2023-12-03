# Neural Linguistic Steganography

Here is the adaption of the original implementation of [harvardnlp/NeuralSteganography](https://github.com/harvardnlp/NeuralSteganography).

This method only use **pretrained GPT2** and requires **no training**. In the original paper, only the first three sentences of the covertext is used as the context for conditional text generation. The secret bits are then fed to the text generation for stego-text.

We DO NOT use the bit encoding of the original repo, since we already have the encoded bits.
