import sentencepiece as spm

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.Train(
    '--input=multilingual_corpus.txt --model_prefix=multi_tokenizer --vocab_size=8000 --model_type=bpe'
)

print("Tokenizer training completed.")
