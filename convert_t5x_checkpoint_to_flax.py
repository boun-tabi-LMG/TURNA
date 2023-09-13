# https://gist.github.com/stefan-it/30e4998ef159f33696e377a46f699d9f

import argparse

from t5x import checkpoints
from transformers import T5Config, FlaxT5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch


def convert_t5x_checkpoint_to_flax(t5x_checkpoint_path, config_name, flax_dump_folder_path):
    config = T5Config.from_pretrained(config_name)
    flax_model = FlaxT5ForConditionalGeneration(config=config)
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)

    split_mlp_wi = "wi_0" in t5x_model["target"]["encoder"]["layers_0"]["mlp"]

    # Encoder
    for layer_index in range(config.num_layers):
        layer_name = f"layers_{str(layer_index)}"
        
        # Self-Attention
        t5x_attention_key = t5x_model["target"]["encoder"][layer_name]["attention"]["key"]["kernel"]
        t5x_attention_out = t5x_model["target"]["encoder"][layer_name]["attention"]["out"]["kernel"]
        t5x_attention_query = t5x_model["target"]["encoder"][layer_name]["attention"]["query"]["kernel"]
        t5x_attention_value = t5x_model["target"]["encoder"][layer_name]["attention"]["value"]["kernel"]

        ## Layer Normalization
        t5x_attention_layer_norm = t5x_model["target"]["encoder"][layer_name]["pre_attention_layer_norm"]["scale"]
        
        if split_mlp_wi:
            t5x_mlp_wi_0 = t5x_model["target"]["encoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            t5x_mlp_wi_1 = t5x_model["target"]["encoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            t5x_mlp_wi = t5x_model["target"]["encoder"][layer_name]["mlp"]["wi"]["kernel"]
        
        t5x_mlp_wo = t5x_model["target"]["encoder"][layer_name]["mlp"]["wo"]["kernel"]
        
        ## Layer Normalization
        t5x_mlp_layer_norm = t5x_model["target"]["encoder"][layer_name]["pre_mlp_layer_norm"]["scale"]
        
        # Assigning
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"]["kernel"] = t5x_attention_key
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"]["kernel"] = t5x_attention_out
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"]["kernel"] = t5x_attention_query
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"]["kernel"] = t5x_attention_value
        
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"]["weight"] = t5x_attention_layer_norm
        
        if split_mlp_wi:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_0"]["kernel"] = t5x_mlp_wi_0
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_1"]["kernel"] = t5x_mlp_wi_1
        else:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi"]["kernel"] = t5x_mlp_wi

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wo"]["kernel"] = t5x_mlp_wo
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"]["weight"] = t5x_mlp_layer_norm

    # Only for layer 0:
    t5x_encoder_rel_embedding = t5x_model["target"]["encoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"]["embedding"] = t5x_encoder_rel_embedding

    # Assigning
    t5x_encoder_norm = t5x_model["target"]["encoder"]["encoder_norm"]["scale"]
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = t5x_encoder_norm

    # Decoder
    for layer_index in range(config.num_decoder_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        t5x_attention_key = t5x_model["target"]["decoder"][layer_name]["self_attention"]["key"]["kernel"]
        t5x_attention_out = t5x_model["target"]["decoder"][layer_name]["self_attention"]["out"]["kernel"]
        t5x_attention_query = t5x_model["target"]["decoder"][layer_name]["self_attention"]["query"]["kernel"]
        t5x_attention_value = t5x_model["target"]["decoder"][layer_name]["self_attention"]["value"]["kernel"]

        ## Layer Normalization
        t5x_pre_attention_layer_norm = t5x_model["target"]["decoder"][layer_name]["pre_self_attention_layer_norm"]["scale"]

        # Encoder-Decoder-Attention
        t5x_enc_dec_attention_key = t5x_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["key"]["kernel"]
        t5x_enc_dec_attention_out = t5x_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["out"]["kernel"]
        t5x_enc_dec_attention_query = t5x_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["query"]["kernel"]
        t5x_enc_dec_attention_value = t5x_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["value"]["kernel"]

        ## Layer Normalization
        t5x_cross_layer_norm = t5x_model["target"]["decoder"][layer_name]["pre_cross_attention_layer_norm"]["scale"]

        # MLP
        if split_mlp_wi:
            t5x_mlp_wi_0 = t5x_model["target"]["decoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            t5x_mlp_wi_1 = t5x_model["target"]["decoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            t5x_mlp_wi = t5x_model["target"]["decoder"][layer_name]["mlp"]["wi"]["kernel"]

        t5x_mlp_wo = t5x_model["target"]["decoder"][layer_name]["mlp"]["wo"]["kernel"]

        ## Layer Normalization
        tx5_mlp_layer_norm = t5x_model["target"]["decoder"][layer_name]["pre_mlp_layer_norm"]["scale"]
        
        # Assigning
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"]["kernel"] = t5x_attention_key
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"]["kernel"] = t5x_attention_out
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"]["kernel"] = t5x_attention_query
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"]["kernel"] = t5x_attention_value

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"]["weight"] = t5x_pre_attention_layer_norm
        
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["k"]["kernel"] = t5x_enc_dec_attention_key
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["o"]["kernel"] = t5x_enc_dec_attention_out
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["q"]["kernel"] = t5x_enc_dec_attention_query
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["v"]["kernel"] = t5x_enc_dec_attention_value
        
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"]["weight"] = t5x_cross_layer_norm
        
        if split_mlp_wi:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_0"]["kernel"] = t5x_mlp_wi_0
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_1"]["kernel"] = t5x_mlp_wi_1
        else:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi"]["kernel"] = t5x_mlp_wi

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wo"]["kernel"] = t5x_mlp_wo
        
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["layer_norm"]["weight"] = tx5_mlp_layer_norm

    # Decoder Normalization
    tx5_decoder_norm = t5x_model["target"]["decoder"]["decoder_norm"]["scale"]
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = tx5_decoder_norm

    # Only for layer 0:
    t5x_decoder_rel_embedding = t5x_model["target"]["decoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"]["embedding"] = t5x_decoder_rel_embedding

    # Token Embeddings
    tx5_token_embeddings = t5x_model["target"]["token_embedder"]["embedding"]
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # LM Head
    flax_model.params["lm_head"]["kernel"] = t5x_model["target"]["decoder"]["logits_dense"]["kernel"]

    flax_model.save_pretrained(flax_dump_folder_path)
    print("T5X Model was sucessfully converted!")

def convert_flax_to_pytorch(flax_dump_folder_path, pytorch_dump_folder_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(flax_dump_folder_path, from_flax=True, torch_dtype=torch.float32)
    model.save_pretrained(pytorch_dump_folder_path)
    print("Flax model was sucessfully converted to Pytorch!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path the TX5 checkpoint."
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=True, help="Config name of T5 model."
    )
    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    args = parser.parse_args()
    convert_t5x_checkpoint_to_flax(args.t5x_checkpoint_path, args.config_name, args.flax_dump_folder_path)
    convert_flax_to_pytorch(args.flax_dump_folder_path, args.flax_dump_folder_path)
    