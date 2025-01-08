import random
import numpy as np
import scipy.stats as stats
import scipy.io as sio
def calculate_similarity(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    num_lines = min(len(lines1), len(lines2))
    random_lines = random.sample(range(num_lines), k=min(num_lines, 1000))  # 随机选择1000行进行计算

    num_similar_lines = 0
    for i in random_lines:
        last_field1 = lines1[i].strip().split()[-1]
        last_field2 = lines2[i].strip().split()[-1]
        if last_field1 == last_field2:
            num_similar_lines += 1

    similarity_ratio = num_similar_lines / len(random_lines) * 100
    return similarity_ratio


file_paths = ['2171trials_humans.txt', '2171trials_ChatGPT_3.5.txt', '2171trials_GPT_4.txt']

similarities = []

for _ in range(5):
    similarity_values = []
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            file1 = file_paths[i]
            file2 = file_paths[j]
            similarity_ratio = calculate_similarity(file1, file2)
            similarity_values.append(similarity_ratio)
    similarities.append(similarity_values)

similarities = np.array(similarities)
mean_similarity = np.mean(similarities, axis=0)
std_similarity = np.std(similarities, axis=0)
confidence_interval = stats.t.interval(0.95, len(similarities[0])-1, loc=mean_similarity, scale=std_similarity/np.sqrt(len(similarities)))
deviation = confidence_interval[1] - mean_similarity

print("mean_similarity:", mean_similarity)
print("95% CI deviation:", deviation)
        
sio.savemat('similarities.mat', {'similarities': similarities[:,0:2]})