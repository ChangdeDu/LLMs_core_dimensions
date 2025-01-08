
def remove_quotes(text):
    if "'" in text:
        text = text.replace("'", "")
    if '"' in text:
        text = text.replace('"', "")
    return text

with open('data/things_concepts.tsv', 'r') as input_file, open('data/variables/image_descriptions_meta.txt', 'w') as output_file:
    for i, line in enumerate(input_file):
        if i >= 1 and i <= 1855:
            columns = line.split('\t')
            col1, col18, col24 = columns[0], remove_quotes(columns[17]), columns[23]
            output_line = f"{i - 1} {col1} ({col24})\n\n"
            output_file.write(output_line)

with open('data/things_concepts.tsv', 'r') as input_file, open('extract_feature_and_dimension_rating/image_descriptions_meta_without_number_space.txt', 'w') as output_file:
    for i, line in enumerate(input_file):
        if i >= 1 and i <= 1855:
            columns = line.split('\t')
            col1, col18, col24 = columns[0], remove_quotes(columns[17]), columns[23]
            output_line = f"{col1} ({col24})\n"
            output_file.write(output_line)