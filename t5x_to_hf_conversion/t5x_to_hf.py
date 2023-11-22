import argparse

from conversion_utils import convert_t5x_checkpoint_to_pytorch

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--t5x_checkpoint_path",
        type=str,
        action="store")
        
    parser.add_argument(
        "--config_path",
        type=str,
        action="store")
        
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        action="store")
        
    args = parser.parse_args()

    t5x_checkpoint_path = args.t5x_checkpoint_path
    config_path = args.config_path
    pytorch_dump_path = args.pytorch_dump_path

    is_encoder_only = False

    convert_t5x_checkpoint_to_pytorch(
            t5x_checkpoint_path, config_path, pytorch_dump_path, is_encoder_only
        )
        
    print('T5X model is successfuly converted to PyTorch model.')
        
if __name__ == "__main__":
    main()