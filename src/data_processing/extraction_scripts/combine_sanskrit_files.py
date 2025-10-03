import os

input_folder = 'C:/Users/soham/Downloads/Sanskrit/Sankrit_Corpus'  # Path to Sanskrit text files folder
output_file = 'sanskrit_corpus.txt'

file_list = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

with open(output_file, 'w', encoding='utf-8') as out_f:
    for filename in file_list:
        path = os.path.join(input_folder, filename)
        with open(path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                out_f.write(line.strip() + '\n')

print(f"Combined {len(file_list)} Sanskrit files into {output_file}")
