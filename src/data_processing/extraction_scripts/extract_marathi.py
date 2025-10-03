import pandas as pd

# Load the Marathi train.csv file
df = pd.read_csv('train.csv')

# Assuming the Marathi text is in the first column (adjust if different)
texts = df.iloc[:, 0].astype(str).tolist()

# Save extracted lines to a text file
with open('marathi_corpus.txt', 'w', encoding='utf-8') as f:
    for line in texts:
        f.write(line.strip() + '\n')

print("Marathi corpus saved to marathi_corpus.txt")
