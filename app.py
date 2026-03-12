import tiktoken
import torch
import chainlit


from save_model import GPTModel
from save_model import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    checkpoint = torch.load("gpt2-medium355M-sft.pth", weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
