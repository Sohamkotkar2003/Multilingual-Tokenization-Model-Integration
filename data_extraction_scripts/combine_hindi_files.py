import os

# Folder containing Hindi text files
input_folder = 'C:/Users/soham/Downloads/Hindi/train/train' # Change this to your Hindi files folder path

# Output file to create
output_file = 'hindi_corpus.txt'

# Get list of text files in the input folder
file_list = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

with open(output_file, 'w', encoding='utf-8') as out_f:
    for filename in file_list:
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                # Write each line to output with newline stripped and re-added cleanly
                out_f.write(line.strip() + '\n')

print(f"Combined {len(file_list)} Hindi files into {output_file}")
