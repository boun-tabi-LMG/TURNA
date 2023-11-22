Converts T5X checkpoint to Huggingface model.

```
python t5x_to_hf.py --t5x_checkpoint_path checkpoint_60000 --pytorch_dump_path converted_pt_model --config_path config.json
```

I decided not to write a function that converts config.gin to config.json for two reasons:
- Doing this manual is much faster.
- We need to do this ONCE only.

Therefore I include config.json, too.