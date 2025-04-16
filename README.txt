=======================================
Aspect-Term Polarity Classification
CentraleSupélec – NLP Project 2025
=======================================

Contributors:
-------------
- Ethan BITAN
- Ismail HATIM
- Antonin SOULIER

Classifier Description:
------------------------

Model used:
- microsoft/deberta-v3-base

Type of classifier:
- Encoder-only transformer model fine-tuned for 3-way sentence classification.

Task formulation:
- Each example is a triplet (sentence, term, aspect_category).
- The model predicts the polarity (positive, negative, neutral) of the opinion expressed about the term in the context of the aspect.

Input and representation:
- Each input is tokenized as:  
  `[CLS] sentence [SEP] term aspect [SEP]`
- We use the tokenizer from Huggingface's `deberta-v3-base`, with truncation and padding to 256 tokens max length.
- The aspect category and target term are added explicitly to the second segment of the input to guide the model’s attention.

Training setup:
- Fine-tuning is done using Huggingface's `Trainer` API.
- Optimizer: AdamW (default settings)
- Epochs: 5
- Batch size: 8
- Device: CPU
- Mixed precision (`fp16`)

Prediction:
- The `predict()` method returns a list of sentiment labels corresponding to the input examples, in order.

Resources used:
- All resources are restricted to the libraries and models authorized in the project guidelines.
- No additional external libraries were used outside of:
  - transformers==4.50.3
  - peft==0.15.1
  - trl==0.16.0
  - datasets==3.5.0
  - sentencepiece==0.2.0
  - lightning==2.5.1
  - ollama==0.4.7
  - pyrallis==0.3.1

Accuracy:
---------
- Accuracy on `devdata.csv`:
  1.2. Eval on the dev set... Acc.: 85.11
  2.2. Eval on the dev set... Acc.: 84.04
  3.2. Eval on the dev set... Acc.: 85.11
  4.2. Eval on the dev set... Acc.: 84.84