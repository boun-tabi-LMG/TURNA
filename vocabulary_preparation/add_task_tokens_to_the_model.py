"""add task tokens"""
from transformers.utils import sentencepiece_model_pb2 as model

m = model.ModelProto()

m.ParseFromString(open("./SentencePiece_32k_Tokenizer.model", "rb").read())

R_DENOISER_TOKEN_PREFIX = "[NLU]"
X_DENOISER_TOKEN_PREFIX = "[NLG]"
S_DENOISER_TOKEN_PREFIX = "[S2S]"


def replace_tokens(m):
    """the following code changes the last three tokens to be the denoiser tokens"""
    least_frequent = sorted(m.pieces, key=lambda x: x.score)[:3]
    # replace the least frequent three tokens with the denoiser tokens
    for i, p in enumerate(least_frequent):
        if i == 0:
            p.piece = R_DENOISER_TOKEN_PREFIX
        elif i == 1:
            p.piece = X_DENOISER_TOKEN_PREFIX
        elif i == 2:
            p.piece = S_DENOISER_TOKEN_PREFIX
        else:
            break


def make_changes(m):
    """add denoiser tokens to the vocabulary and update the first three tokens to be
    the ID=0 for padding, ID=1 for EOS, and ID=2 for UNK.
    """
    replace_tokens(m)
    # update the first three tokens to be the ID=0 for padding, ID=1 for EOS, and ID=2 for UNK
    for i, p in enumerate(m.pieces):
        if i == 0:
            p.piece = "<pad>"
        elif i == 1:
            p.piece = "</s>"
        elif i == 2:
            p.piece = "<unk>"
        elif i == 3:
            p.piece = "<s>"
        else:
            break


def load_and_list_tokens_from_a_vocabulary(vocabulary_filepath):
    """this function uses X library"""
    m = model.ModelProto()

    m.ParseFromString(open(vocabulary_filepath, "rb").read())

    for idx, p in enumerate(m.pieces):
        yield idx, p.piece, p.score


make_changes(m)

with open("SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model", "wb") as f:
    f.write(m.SerializeToString())

for (
    i,
    value,
    score,
) in load_and_list_tokens_from_a_vocabulary(
    "SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model"
):
    print(f"Token: {i} -- {value} -- {score}")
