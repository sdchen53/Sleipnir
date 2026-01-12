from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # You can use other models too

# Sentence to be tokenized
sentence = "I love natural language processing!"

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)  # Get tokenized words/subwords
token_ids = tokenizer(sentence)  # Get full tokenizer output including token IDs, etc.

# Print results
print("Tokens:", tokens)  # Tokenized result
print("Token IDs:", token_ids['input_ids'])  # Numeric representation of tokens