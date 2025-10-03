import pandas as pd

# Update filename to your large English CSV/text file path
df = pd.read_csv('cefr_leveled_texts.csv')

# Adjust column name containing English text
texts = df['text'].astype(str).tolist()

with open('english_corpus.txt', 'w', encoding='utf-8') as f:
    for line in texts:
        f.write(line.strip() + '\n')

print("English corpus saved to english_corpus.txt")
