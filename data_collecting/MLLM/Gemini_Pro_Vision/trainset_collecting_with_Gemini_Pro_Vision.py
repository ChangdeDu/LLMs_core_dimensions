from picture_joint import picture_joint
import time
import csv
import os
from PIL import Image
import google.generativeai as genai
from skimage import io
import PIL

def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)

def write_trial_data_to_csv(filename, triplets, prompts, responses):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(triplets, prompts, responses))

# Define API request parameters
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
GOOGLE_API_KEY = 'Fill in your GOOGLE_API_KEY'
genai.configure(api_key=GOOGLE_API_KEY)
model_engine = "gemini-pro"

temperature = 0.01  # The larger the value, the more random the response
max_tokens = 2048

filename = 'data/MLLMs/Gemini_Pro_Vision/triplet_dataset/trainset_Gemini_Pro_Vision.txt'
if not os.path.exists(filename):
    with open(filename, 'w') as file:
        pass
line_count = count_lines(filename)

trainset_humans = 'data/Humans/triplet_dataset/trainset.txt'
total_lines = count_lines(trainset_humans)

# Define the range of trials
start_trial = line_count
end_trial = total_lines

# Read selected line numbers from humans' txt file
with open(trainset_humans,
          'r') as f:
    human_records = f.read().splitlines()

for trial in range(start_trial, end_trial):
    start_time = time.time()  # Record start time
    print('####################################################################################### saved_id = ', trial)

    line = human_records[trial]
    split_line = line.split(' ')
    print('human: ', split_line)

    img_path1 = "./ori_behav_stimuli/image_" + str(int(split_line[0]) + 1) + '_ori.jpg'
    img_path2 = "./ori_behav_stimuli/image_" + str(int(split_line[1]) + 1) + '_ori.jpg'
    img_path3 = "./ori_behav_stimuli/image_" + str(int(split_line[2]) + 1) + '_ori.jpg'

    # The prompt we used
    prompt = "You are shown three object images side by side and are asked to report the image that was the least similar to the other two. \
    You should focus your judgement on the object, but you are not given additional constraints as to the strategy you should use. \
    If you did not recognize the object, you should base your judgement on your best guess of what the object could be. 1. Tell me your answer. 2. Tell me the location of the object you have chosen. 3. Explain the reasons."  # Your answer shoude in the format that 'The image in the xxx (left, middle, right) is the least similar to the other two. The specific explanation...'"

    config = {"max_output_tokens": max_tokens, "temperature": temperature, "top_p": 1, "top_k": 32}
    safety_settings = {
        "harm_category_dangerous_content": "block_none",
        "harm_category_harassment": "block_none",
        "harm_category_hate_speech": "block_none",
        "harm_category_sexually_explicit": "block_none"
    }

    img = picture_joint(img_path1, img_path2, img_path3)
    img = Image.fromarray(img)  # Convert to a PIL image object
    # Send an HTTP POST request to get the API response
    model = genai.GenerativeModel('gemini-pro-vision')
    response_raw = model.generate_content([prompt, img], generation_config=config, stream=True,
                                          safety_settings=safety_settings)

    response_raw.resolve()
    Response = response_raw.text

    # Output the model's response and find the image number matching that description
    response = Response.lower()
    if "left" in response.split('\n')[1].split('.')[1]:
        sorted_images = [split_line[1], split_line[2], split_line[0]]

    elif "middle" in response.split('\n')[1].split('.')[1] or "center" in response.split('\n')[1].split('.')[1]:
        sorted_images = [split_line[0], split_line[2], split_line[1]]

    elif "right" in response.split('\n')[1].split('.')[1]:
        sorted_images = [split_line[0], split_line[1], split_line[2]]

    # Write prompts and responses to the CSV file
    group_prompts = [None] * 1
    group_responses = [None] * 1
    group_results = [None] * 1
    group_prompts[0] = prompt
    group_responses[0] = Response
    group_results[0] = sorted_images
    write_trial_data_to_csv('data/MLLMs/Gemini_Pro_Vision/triplet_dataset/prompts&responses_trainset_Gemini_Pro_Vision.csv', group_results,
                            group_prompts, group_responses)

    print("model:", sorted_images)
    # Save the sorted image numbers to a txt file
    with open('data/MLLMs/Gemini_Pro_Vision/triplet_dataset/trainset_Gemini_Pro_Vision.txt', 'a') as f:
        f.write(' '.join(sorted_images) + '\n')

    # Print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time} seconds")

print("Program execution complete")
