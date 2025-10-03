corpus_files = {
    'marathi': 'marathi_corpus.txt',
    'hindi': 'hindi_corpus.txt',
    'sanskrit': 'sanskrit_corpus.txt',
    'english': 'english_corpus.txt'
}

output_file = 'multilingual_corpus.txt'

with open(output_file, 'w', encoding='utf-8') as outfile:
    for lang, filepath in corpus_files.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    clean_line = line.strip()
                    if clean_line:
                        outfile.write(clean_line + '\n')
            print(f"{lang} corpus added successfully.")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found and skipped.")

print(f"Combined multilingual corpus saved as {output_file}")
