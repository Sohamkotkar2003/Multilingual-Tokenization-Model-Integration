from transformers import AutoTokenizer

# Load a pre-trained tokenizer (multilingual)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# Sample sentence in Hindi (Devanagari script)
sentence = "यह एक परीक्षण वाक्य है।"

# Tokenize
tokens = tokenizer.tokenize(sentence)

print("Tokens:", tokens)
