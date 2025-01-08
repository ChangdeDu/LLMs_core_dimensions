import requests
import random
import time
import csv
import os

def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)

def write_trial_data_to_csv(filename, triplets, prompts, responses):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(triplets, prompts, responses))

# Define API request parameters
openai_api_key = "Fill in your openai api_key"
model_engine = "gpt-3.5-turbo"
temperature = 0.01 # The larger the value, the more random the response
max_tokens = 1024
n = 1
count = 0

filename = 'data/LLMs/ChatGPT-3.5/triplet_dataset/full_sampling_for_48_objects_ChatGPT-3.5.txt'
if not os.path.exists(filename):
    with open(filename, 'w') as file:
        pass
line_count = count_lines(filename)

full_sampling = "data/LLMs/ChatGPT-3.5/triplet_dataset/full_sampling_for_48_objects_random.txt"
total_lines = count_lines(full_sampling)

# Read image descriptions and numbers from a txt file, which was saved in advance.
with open('data/LLMs/ChatGPT-3.5/image_descriptions_meta.txt', 'r') as f:
    descriptions = f.read().split('\n\n')

# Remove trailing newline characters from each image description string and split into number and description parts
descriptions = [desc.strip().split(' ', 1) for desc in descriptions]

# Read selected line numbers from humans' txt file
with open(full_sampling,
          'r') as f:
    human_records = f.read().splitlines()

# Define the range of trials
start_trial = line_count
end_trial = total_lines
for trial in range(start_trial,end_trial):
    start_time = time.time()  # Record start time
    print('####################################################################################### saved_id = ', trial)
    line = human_records[trial]
    split_line = line.split(' ')
    print('human: ', split_line)
    # Convert selected line numbers to integers and select corresponding image descriptions
    selected_descriptions = [descriptions[int(row)] for row in split_line]

    # The prompt structure we used
    prompt = "Given a triplet of objects {'[IMAGE_A]', '[IMAGE_B]', '[IMAGE_C]'}, which one in the triplet is the odd-one-out? Please give the answer first and then explain in detail."

    # [IMAGE_A], [IMAGE_B] and [IMAGE_C] were replaced with the respective object descriptions
    prompt = prompt.replace("[IMAGE_A]", selected_descriptions[0][1])
    prompt = prompt.replace("[IMAGE_B]", selected_descriptions[1][1])
    prompt = prompt.replace("[IMAGE_C]", selected_descriptions[2][1])

    # Send an HTTP POST request to the OpenAI API to get the API response
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {openai_api_key}"}

    data = {
        "model": model_engine,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,

    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response_json = response.json()
    response = response_json["choices"][0]["message"]["content"].strip()
    odd_one_out = response.lower()

    # Output the model's response and find the image number matching that description
    if selected_descriptions[0][1].split(' ')[0] in odd_one_out.split('\n\n')[0][0:100]:
        selected_image = selected_descriptions[0][0]
        sorted_images = [selected_descriptions[1][0], selected_descriptions[2][0], selected_image]

    elif selected_descriptions[1][1].split(' ')[0] in odd_one_out.split('\n\n')[0][0:100]:
        selected_image = selected_descriptions[1][0]
        sorted_images = [selected_descriptions[0][0], selected_descriptions[2][0], selected_image]

    elif selected_descriptions[2][1].split(' ')[0] in odd_one_out.split('\n\n')[0][0:100]:
        selected_image = selected_descriptions[2][0]
        sorted_images = [selected_descriptions[0][0], selected_descriptions[1][0], selected_image]

    # Write prompts and responses to the CSV file
    group_prompts = [None] * 1
    group_responses = [None] * 1
    group_results = [None] * 1
    group_prompts[0] = prompt
    group_responses[0] = response
    group_results[0] = sorted_images
    write_trial_data_to_csv('data/LLMs/ChatGPT-3.5/triplet_dataset/prompts&responses_full_sampling_for_48_objects_ChatGPT-3.5.csv', group_results, group_prompts, group_responses)

    print("model:", sorted_images)
    # Save the sorted image numbers to a txt file
    with open('data/LLMs/ChatGPT-3.5/triplet_dataset/full_sampling_for_48_objects_ChatGPT-3.5.txt', 'a') as f:
        f.write(' '.join(sorted_images) + '\n')

    # Print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time} seconds")

print("Program execution complete")
