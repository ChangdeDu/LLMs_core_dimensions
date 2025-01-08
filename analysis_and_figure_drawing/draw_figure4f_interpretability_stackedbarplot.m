clear all;
% run this script from where it is located
base_dir = pwd;

data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5/dim_naming_chatgpt');
csvFilePath = fullfile(data_dir,'automated_dimension_analysis_visual_vs_semantic_for_chatgpt.csv');
dataTable = readtable(csvFilePath);
columnData = dataTable{:, 5};
count1_chatgpt = sum(columnData == 1);
count2_chatgpt = sum(columnData == 0);

data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision/dim_naming_gemini');
csvFilePath = fullfile(data_dir,'automated_dimension_analysis_visual_vs_semantic_for_gemini.csv');
dataTable = readtable(csvFilePath);
columnData = dataTable{:, 5};
count1_gemini = sum(columnData == 1);
count2_gemini = sum(columnData == 0);

data_dir = fullfile(base_dir,'data/Humans/dim_naming_human');
csvFilePath = fullfile(data_dir,'automated_dimension_analysis_visual_vs_semantic_for_human.csv');
dataTable = readtable(csvFilePath);
columnData = dataTable{:, 5};
count1_human = sum(columnData == 1);
count2_human = sum(columnData == 0);

% 数据
labels = {'LLM', 'MLLM', 'Human'};
data_easy = [count1_chatgpt, count1_gemini, count1_human];
data_hard = [count2_chatgpt, count2_gemini, count2_human];
% 计算每一组数据的百分比
data_sum = data_easy + data_hard;
data1_percent = data_easy ./ data_sum;
data2_percent = data_hard ./ data_sum;

% 创建一个新的图像窗口
fig = figure('Position',[1500 1500 1500 1500],'color','none');

% 设置颜色映射
colormap(lines(3));

% 绘制百分比堆积柱形图
barWidth = 0.5; % 设置柱子宽度
xPositions = [0.2, 0.5, 0.8]; % 设置柱子位置
h = bar(xPositions, [data1_percent; data2_percent], 'stacked', 'BarWidth', barWidth);

% 设置每一组堆积柱形图的颜色
h(1).FaceColor = [0.40 0.76 0.60];
h(2).FaceColor = [0.55 0.63 0.80];

% 将y轴范围设置
ylim([0 1.02]);
xlim([0 1.0]);

% 设置x轴的刻度和标签
set(gca, 'XTick', xPositions);
set(gca, 'XTickLabel', labels);

% 添加图例和标签
legend('Easy', 'Hard', 'Location', 'best',  'FontSize', 35);
ylabel('Percentage', 'FontSize', 38);
title('Interpretability', 'FontSize', 32);

hax = gca;
set(gca,'FontSize',35) 
hax.TickDir = 'both';
% hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1.5;
hax.Box = 'off';

set(fig, 'Position', [100, 100, 600, 900]); % [left, bottom, width, height]
exportgraphics(fig, 'interpretable_easy_vs_hard_barplot.pdf', 'ContentType', 'vector');
close(fig);