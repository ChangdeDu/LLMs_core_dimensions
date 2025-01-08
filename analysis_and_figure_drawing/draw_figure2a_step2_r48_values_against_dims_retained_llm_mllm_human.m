
% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load chatgpt relevant data
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
spose_embedding_aug = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
spose_embedding = [spose_embedding spose_embedding_aug(:,67:73)];

% load 48 object RDM
load(fullfile(data_dir,'RDM48_triplet.mat'));
%% Load smaller version of words for each image
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');

% Initialize variables for correlation values and dimensions
dims_to_check_chatgpt = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_chatgpt = zeros(length(dims_to_check_chatgpt), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_chatgpt
    % Retain the first 'd' dimensions of the embedding
    spose_embedding_d = spose_embedding(:, 1:d);
    
    % Calculate the dot product and similarity matrix
    dot_product_d = spose_embedding_d * spose_embedding_d';
    esim_d = exp(dot_product_d);
    
    % Compute proximity (similarity) for 48 objects
    cp = zeros(1854, 1854);
    ctmp = zeros(1, 1854);
    for i = 1:1854
        for j = i+1:1854
            ctmp = zeros(1, 1854);
            for k_ind = 1:length(wordposition48)
                k = wordposition48(k_ind);
                if k == i || k == j, continue, end
                ctmp(k) = esim_d(i,j) / (esim_d(i,j) + esim_d(i,k) + esim_d(j,k));
            end
            cp(i,j) = sum(ctmp);
        end
    end
    cp = cp / 48;
    cp = cp + cp';
    cp(logical(eye(size(cp)))) = 1;

    spose_sim48_d = cp(wordposition48, wordposition48);
    
    % Calculate the correlation r48 for current dimension
    r48_values_chatgpt(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end




%% Load gemini relevant data
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
spose_embedding_aug = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
spose_embedding = [spose_embedding spose_embedding_aug(:,67:73)];
% load 48 object RDM
load(fullfile(data_dir,'RDM48_triplet.mat'));
%% Load smaller version of words for each image
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');

% Initialize variables for correlation values and dimensions
dims_to_check_gemini = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_gemini = zeros(length(dims_to_check_gemini), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_gemini
    % Retain the first 'd' dimensions of the embedding
    spose_embedding_d = spose_embedding(:, 1:d);
    
    % Calculate the dot product and similarity matrix
    dot_product_d = spose_embedding_d * spose_embedding_d';
    esim_d = exp(dot_product_d);
    
    % Compute proximity (similarity) for 48 objects
    cp = zeros(1854, 1854);
    ctmp = zeros(1, 1854);
    for i = 1:1854
        for j = i+1:1854
            ctmp = zeros(1, 1854);
            for k_ind = 1:length(wordposition48)
                k = wordposition48(k_ind);
                if k == i || k == j, continue, end
                ctmp(k) = esim_d(i,j) / (esim_d(i,j) + esim_d(i,k) + esim_d(j,k));
            end
            cp(i,j) = sum(ctmp);
        end
    end
    cp = cp / 48;
    cp = cp + cp';
    cp(logical(eye(size(cp)))) = 1;

    spose_sim48_d = cp(wordposition48, wordposition48);

    % Calculate the correlation r48 for current dimension
    r48_values_gemini(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end




%% Load human relevant data
data_dir = fullfile(base_dir,'data/Humans/');
% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));
% load 48 object RDM
load(fullfile(data_dir,'RDM48_triplet.mat'));
%% Load smaller version of words for each image
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');

% Initialize variables for correlation values and dimensions
dims_to_check_human = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_human = zeros(length(dims_to_check_human), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_human
    % Retain the first 'd' dimensions of the embedding
    spose_embedding_d = spose_embedding(:, 1:d);
    
    % Calculate the dot product and similarity matrix
    dot_product_d = spose_embedding_d * spose_embedding_d';
    esim_d = exp(dot_product_d);
    
    % Compute proximity (similarity) for 48 objects
    cp = zeros(1854, 1854);
    ctmp = zeros(1, 1854);
    for i = 1:1854
        for j = i+1:1854
            ctmp = zeros(1, 1854);
            for k_ind = 1:length(wordposition48)
                k = wordposition48(k_ind);
                if k == i || k == j, continue, end
                ctmp(k) = esim_d(i,j) / (esim_d(i,j) + esim_d(i,k) + esim_d(j,k));
            end
            cp(i,j) = sum(ctmp);
        end
    end
    cp = cp / 48;
    cp = cp + cp';
    cp(logical(eye(size(cp)))) = 1;

    spose_sim48_d = cp(wordposition48, wordposition48);
    
    % Calculate the correlation r48 for current dimension
    r48_values_human(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end



% Plot the r48 values against the dimensions retained
fig = figure('Position',[800 800 550 500],'color','none');

customColor1 = [.2 .6 .4]; 
customColor2 = [.6 .2 .4]; 
customColor3 = [.2 .4 .6]; 
plot(dims_to_check_chatgpt, r48_values_chatgpt, 'o-', 'Color', customColor1, 'LineWidth', 3.0, 'MarkerSize', 3.5);
hold on;
plot(dims_to_check_gemini, r48_values_gemini, 'o-', 'Color', customColor2, 'LineWidth', 3.0, 'MarkerSize', 3.5);
hold on;
plot(dims_to_check_human, r48_values_human, 'o-', 'Color', customColor3, 'LineWidth', 3.0, 'MarkerSize', 3.5);
line([66 66], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2.0);
legend('LLM', 'MLLM', 'Human', 'Location', 'best');


ylabel('Representational similarity score', 'FontSize', 22);
xlabel('Number of Dimensions Retained', 'FontSize', 22);
title('Prediction of measured RSM', 'FontSize', 20);
xlim([1,75])
ylim([0.5 0.95])
set (gca,'xtick',(1 : 10 :75));
set (gca,'Ytick',(0.5 : 0.05 :0.9));

hax = gca;
set(gca,'FontSize',19) 
hax.Box = 'off';
hax.LineWidth = 1.5; 

exportgraphics(fig, 'figures/r48_values_against_dims.pdf', 'ContentType', 'vector');

close(fig);





