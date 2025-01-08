import random
import itertools
import scipy.io as sio

mat_file = "data/wordposition48.mat"
mat_data = sio.loadmat(mat_file)
# Get special numbers
special_numbers = mat_data["wordposition48"] - 1
special_numbers = special_numbers.flatten().tolist()
total_numbers = 1854  # Total number of image identifiers
num_triples = 100000  # Number of triples to generate
output_file = "data/LLMs/ChatGPT-3.5/triplet_dataset/full_sampling_for_48_objects_random.txt"

# Randomly select two numbers from the special numbers
def select_special_numbers():
    return random.sample(special_numbers, 2)

# Randomly select one number from the remaining numbers
def select_regular_number():
    return random.choice(list(set(range(0, total_numbers)) - set(special_numbers)))

# Generate triples and write to file
with open(output_file, 'w') as f:
    for _ in range(num_triples):
        special_1, special_2 = select_special_numbers()
        regular = select_regular_number()
        triple = [special_1, special_2, regular]
        line = ' '.join(map(str, triple))
        f.write(line + '\n')

print("done！")
