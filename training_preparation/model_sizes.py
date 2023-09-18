"""Model sizes"""

from transformers import TFT5ForConditionalGeneration, T5Config

import pandas as pd

df = pd.read_csv(
    "./training_preparation/Turkish LLM - Model sizes and memory requirements - Sheet1 - 02.tsv",
    sep="\t",
)

df = df.drop(columns=["name", "model_size", "model_size_in_bytes"])


def perform_casting(df):
    """for every column, cast float values to int"""
    for column in df.columns:
        try:
            df[column] = df[column].astype(int)
        except ValueError as ex:
            print(ex)
    return df


df = perform_casting(df)
print(df)
print(df.dtypes)

results = []

for i, row in df.iterrows():
    row_dict = row.to_dict()
    # print(row_dict)
    # print(row_dict["num_layers"])
    # print(type(row_dict["num_layers"]))

    config = T5Config(
        vocab_size=row_dict["vocab_size"],
        d_model=row_dict["d_model"],
        d_ff=row_dict["d_ff"],
        num_layers=row_dict["num_layers"],
        num_heads=row_dict["num_heads"],
    )

    # config = T5Config(
    #     vocab_size=32000,
    #     d_model=1024,
    #     d_ff=4096,
    #     num_layers=24,
    #     num_heads=16,
    # )

    model = TFT5ForConditionalGeneration(config)
    model.build((row_dict["batch_size"], row_dict["sequence_length"]))
    model_size = model.num_parameters()

    model_size_in_bytes = model_size * 4
    model_size_in_gigabytes = model_size * 4 / 1024 / 1024 / 1024

    results.append((model_size, model_size_in_bytes, model_size_in_gigabytes))

df["model_size"] = [result[0] for result in results]
df["model_size_in_bytes"] = [result[1] for result in results]
df["model_size_in_gigabytes"] = [result[2] for result in results]
df.to_csv("./training_preparation/model_sizes.tsv", sep="\t", index=False)
