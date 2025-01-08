import numpy as np
import os
from PIL import Image, ImageOps

# 读取npz文件
# folder = 'results/chatgpt3.5_things/100d/adam/gaussian/0.25/1.0/0.6/seed42/'
folder = '/data1/home/cddu/pythonProject/code/SPoSE/results/chatgpt/100d/0.004/seed142/'
data = np.load(folder+'weights_sorted.npy')
image_folder = '/nfs/nica-datashop/Things-data/ori_behav_stimuli'
# 读取pruned_loc数组
pruned_loc = data

# 获取数组的维度
dimensions = pruned_loc.shape[1]

# 创建保存结果图片的文件夹
output_folder = folder+'dim_visualization_images'
os.makedirs(output_folder, exist_ok=True)

# 遍历每个维度
for dim in range(dimensions):
    # 获取当前维度的值
    dim_values = pruned_loc[:, dim]

    # 根据值的大小降序排序，返回排序后的索引
    sorted_indices = np.argsort(dim_values)[::-1]

    # 获取前6个索引
    top_indices = sorted_indices[:6]
    # 创建一个新的图片来容纳拼接后的图片
    concat_width = 0
    concat_height = 0
    images_row = []
    for index in top_indices:
        # 读取对应文件夹下的图片
        image_path = image_folder + f'/image_{index+1}_ori.jpg'
        image = Image.open(image_path)

        # 调整图片大小为256x256像素
        resized_image = image.resize((256, 256))

        # 记录图片的宽度和高度
        image_width, image_height = resized_image.size
        concat_width += image_width
        concat_height = max(concat_height, image_height)

        # 添加调整大小后的图片到行列表
        images_row.append(resized_image)

    # 创建新图片，将图片拼接成一行
    concatenated_image = Image.new('RGB', (concat_width, concat_height))
    x_offset = 0
    for image in images_row:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # 添加黑色边框
    bordered_image = ImageOps.expand(concatenated_image, border=5, fill='black')

    # 保存拼接后的图片
    new_image_path = f'{output_folder}/dimension_{dim+1}.jpg'
    bordered_image.save(new_image_path)
    print('saved_dim_id = ', dim)