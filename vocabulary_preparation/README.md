# Tokenizer & Vocabulary Preparation

This directory contains the scripts used to prepare the tokenizers and vocabularies for the datasets used in the project. 

## SentencePiece
`SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model` is the tokenizer used in traning the models. It is based on the `SentencePiece_32k_tokenizer.model` and contains additional task specific tokens. This can be achieved running

```bash
python add_task_tokens_to_the_model.py     
```

## HuggingFace tokenizer

`VBARTTokenizer_T5_Sentinels` contains files for the HuggingFace version of the tokenizer.