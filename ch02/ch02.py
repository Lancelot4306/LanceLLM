import os
import urllib.request
import re



# if not os.path.exists("the-verdict.txt"):
#     url = ("https://raw.githubusercontent.com/rasbt/"
#            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#            "the-verdict.txt")
#     file_path = "the-verdict.txt"
#     urllib.request.urlretrieve(url, file_path)



# open and read the file (dataset)
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# # separate all characters
# result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# # no whitespace
# result = [item.strip() for item in result if item.strip()]
# preprocessed = result

# all_words = sorted(set(preprocessed))

# vocab_size = len(all_words)

# vocab = {token:integer for integer, token in enumerate(all_words)}



# Tokenizer V1
# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}
    
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
        
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text

# # instantiating an object   
# tokenizer = SimpleTokenizerV1(vocab)

# text = """"It's the last he painted, you know," 
#            Mrs. Gisburn said with pardonable pride."""

# # encoding into token ids (str to int)
# ids = tokenizer.encode(text)

# # decoding (int to str)
# print(tokenizer.decode(ids))

# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# vocab = {token:integer for integer,token in enumerate(all_tokens)}

# print(len(vocab.items()))



# Tokenizer V2
# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = { i:s for s,i in vocab.items()}
    
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [
#             item if item in self.str_to_int 
#             else "<|unk|>" for item in preprocessed
#         ]

#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
        
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
#         return text
    
# tokenizer = SimpleTokenizerV2(vocab)

# text = "Hello, do you like tea. Is this-- a test?"

# # encoding into token ids (str to int)
# print(tokenizer.encode(text))

# # decoding (int to str)
# print(tokenizer.decode(tokenizer.encode(text)))



## Byte-Pair Encoding
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# print(tokenizer.encode("Hello world"))
# print(tokenizer.decode(tokenizer.encode("Hello world")))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# enc_text = tokenizer.encode(raw_text)

# enc_sample = enc_text[50:]


# context_size = 4

# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# print(f"x: {x}")
# print(f"y:      {y}")


# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]

#     # print(context, "---->", desired)
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))




## Data sampling with a sliding window
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=4, stride=4, shuffle=False
# )

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)


# dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)




## Token embeddings
# input_ids = torch.tensor([2, 3, 5, 1])

# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# print(embedding_layer.weight)

# print(embedding_layer(torch.tensor([3])))

# print(embedding_layer(input_ids))




## Encoding word positions (positional embeddings)
vocab_size = 50257 # == gpt2 model
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Token embedding: ", token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("Position embedding: ", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("Input(token + pos) embedding: ", input_embeddings.shape)