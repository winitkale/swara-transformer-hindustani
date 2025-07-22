from transformers import GPT2LMHeadModel

def load_model(model_name="gpt2", vocab_size=None):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if vocab_size:
        model.resize_token_embeddings(vocab_size)
    return model
