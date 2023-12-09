import re
import json
import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import tensorflow as tf

from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast


JSON_TEST_DATA_PATH = "melik_custom_noised_test_cases.tar.gz"
TOKENIZER_PATH = 'VBARTTokenizer_T5_Sentinels'


gzip_file = gzip.open(JSON_TEST_DATA_PATH)
byte_lines = [line for line in gzip_file]
json_lines = [json.loads(byte_line) for byte_line in byte_lines]
print('Number of Test Cases:', len(json_lines))

def pretty_print(ground_truth, model_output):
    """
    - For both Ground Truth and Model Output:
        - count the number of <extra_id_{idx}> tokens.
        - iterate them and get the start and end locations of each.
        - print between <extra_id_{idx}> and <extra_id_{idx+1}>
    """

    # Count the number of sentinel tokens
    num_sentinel_tokens = len(re.findall('<extra_id_', model_output))
    sentinel_token_locs = [] #((ground_truth_start, ground_truth_end), (model_output_start, model_output_end))
    # Extract sentinel token locations
    for idx in range(num_sentinel_tokens):
        sentinel_token = f'<extra_id_{idx}>'
        sentinel_token_len = len(sentinel_token)
    
        # Ground Truth
        start_loc_in_ground_truth = ground_truth.find(sentinel_token)
        end_loc_in_ground_truth = start_loc_in_ground_truth + sentinel_token_len
    
        # Model Output
        start_loc_in_model_output = model_output.find(sentinel_token)
        end_loc_in_model_output = start_loc_in_model_output + sentinel_token_len
    
        sentinel_token_locs.append(((start_loc_in_ground_truth, end_loc_in_ground_truth), (start_loc_in_model_output, end_loc_in_model_output)))

    # Print sentinel tokens 
    for idx in range(num_sentinel_tokens - 1):
        ground_truth_print = ground_truth[sentinel_token_locs[idx][0][1]:sentinel_token_locs[idx+1][0][0]]
        model_output_print = model_output[sentinel_token_locs[idx][1][1]:sentinel_token_locs[idx+1][1][0]]
        print(f'<extra_id_{idx}>', ground_truth_print, '|', model_output_print, f'<extra_id_{idx+1}>')
		
		
config_file = 'config.json'
#pytorch_total_params = sum(p.numel() for p in model.parameters())
#print(f'Number of Model Parameters: {pytorch_total_params:,}')
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

checkpoints = [file for file in os.listdir() if "converted_pt_model_" in file]
checkpoints = sorted(checkpoints, key = lambda x: int(x.split('_')[-1]))
print(checkpoints)
print('--------')

for json_line in json_lines:
    model_input = json_line['inputs']
    model_input = model_input.replace(' <extra_id_', '<extra_id_') # Huggingface Tokenizer creates extra whitespace tokens. I prevent them here.
    ground_truth = json_line['targets']
    
    inputs_encoded = tokenizer.encode(model_input)
    model_inputs = tf.keras.utils.pad_sequences([inputs_encoded], maxlen = 512, padding = 'post', truncating = 'post')
    print('Model Input:\n', tokenizer.decode(model_inputs[0]))
    print('------------------------------------')
    model_inputs = torch.from_numpy(model_inputs).to(device)
    for checkpoint in checkpoints:
        print('Checkpoint:', checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        
        model_outputs = model.generate(inputs = model_inputs, max_length = 512, do_sample = False, num_beams = 1)
        model_output_as_list = model_outputs.numpy()[0].tolist()
        model_output = tokenizer.decode(model_output_as_list, skip_special_tokens = False)
        #print(model_output)
        pretty_print(ground_truth, model_output)
        print('=====================================')
    print('\n\n')