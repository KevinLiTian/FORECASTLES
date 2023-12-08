import torch
# from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, BertModel, OpenAIGPTModel

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def encode_text(text, max_len=50):
    """
    Given a list of text, tokenize it and encode it
    :param max_len: Max length of string
    :param text: list[str]
    :return: tensor of hidden state encodings (N x M x 768) and the attention masks (N x M)
            where M is the length of the largest string.
    """
    text.append('A '*max_len)
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = "left"
    # Tokenize input
    tokenized_text = tokenizer(text, return_tensors='pt', padding=True)
    # Load pre-trained model (weights)
    model = OpenAIGPTModel.from_pretrained("openai-gpt")
    outputs = model(input_ids=tokenized_text['input_ids'],
                    attention_mask=tokenized_text['attention_mask'],
                    # pad_token_id=tokenizer.pad_token_id,
                    )
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[:-1], tokenized_text['attention_mask'][:-1]


if __name__ == "__main__":
    last_hidden_states, mask = encode_text(["Who what Jim? Jim was a clown.", "Hello World"])
