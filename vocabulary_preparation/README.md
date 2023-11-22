# Tokenizer & Vocabulary Preparation

This directory contains the scripts used to prepare the tokenizers and vocabularies for the datasets used in the project. 

## SentencePiece
`SentencePiece_32k_tokenizer.model`` is the base tokenizer and `dd_task_tokens_to_the_model.py` adds task-specific tokens to the model.

### Add task-specific tokens to the SentencePiece model

```bash
python add_task_tokens_to_the_model.py     
```

## HuggingFace tokenizer

`VBARTTokenizer_T5_Sentinels` contains files for the HuggingFace version of the tokenizer.
