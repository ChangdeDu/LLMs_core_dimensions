
% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

% load embedding
data_dir = fullfile(base_dir,'data/DNNs/simclr');
% load 48 object RDM
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');

spose_embedding= load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
% load embedding
data_dir = fullfile(base_dir,'data/DNNs/cornet_s');
spose_embedding_cornet_s = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
% load embedding
data_dir = fullfile(base_dir,'data/Humans/');
spose_embedding_human = load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_simclr = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_simclr = zeros(length(dims_to_check_simclr), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_simclr
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
    r48_values_simclr(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end



% load embedding
data_dir = fullfile(base_dir,'data/DNNs/cornet_s');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_cornet_s = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_cornet_s = zeros(length(dims_to_check_cornet_s), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_cornet_s
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
    r48_values_cornet_s(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end



% load embedding
data_dir = fullfile(base_dir,'data/DNNs/alexnet');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_alexnet = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_alexnet = zeros(length(dims_to_check_alexnet), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_alexnet
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
    r48_values_alexnet(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end



% load embedding
data_dir = fullfile(base_dir,'data/DNNs/resnet18');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_resnet18 = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_resnet18 = zeros(length(dims_to_check_resnet18), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_resnet18
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
    r48_values_resnet18(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end




% load embedding
data_dir = fullfile(base_dir,'data/DNNs/vgg16');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_vgg16 = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_vgg16 = zeros(length(dims_to_check_vgg16), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_vgg16
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
    r48_values_vgg16(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end


% load embedding
data_dir = fullfile(base_dir,'data/DNNs/gabor');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_gabor = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_gabor = zeros(length(dims_to_check_gabor), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_gabor
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
    r48_values_gabor(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end


% load embedding
data_dir = fullfile(base_dir,'data/DNNs/CLIPvison_ViT_L14');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_CLIPvison = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_CLIPvison = zeros(length(dims_to_check_CLIPvison), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_CLIPvison
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
    r48_values_CLIPvison(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end

% load embedding
data_dir = fullfile(base_dir,'data/DNNs/CLIPtext_ViT_L14');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));


% Initialize variables for correlation values and dimensions
dims_to_check_CLIPtext = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_CLIPtext = zeros(length(dims_to_check_CLIPtext), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check_CLIPtext
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
    r48_values_CLIPtext(d) = corr(squareformq(spose_sim48_d), squareformq(1 - RDM48_triplet));
end

% Plot the r48 values against the dimensions retained
fig = figure('Position',[800 800 900 500],'color','none');

colors = [
    [.2 .6 .4];
    [.6 .2 .4]; 
    [.2 .4 .6];
    [.8 .2 .4];
    [.4 .2 .8]; 
    [.8 .5 .3]; 
    [.2 .3 .6];
    [.3 .7 .7]; 
    [.7 .1 .6]; 
    [.2 .4 .4]; 
    [.5 .7 .5]; 
    [.9 .1 .6]; 
    [.5 .1 .6]; 
    [.2 .5 .4]
];

customColor1 = colors(10,:); 
customColor2 = colors(11,:); 
customColor3 = colors(12,:); 
customColor4 = colors(7,:); 
customColor5 = colors(8,:); 
customColor6 = colors(13,:); 
customColor7 = colors(5,:); 
customColor8 = colors(6,:); 



plot(dims_to_check_simclr, r48_values_simclr, 'o-', 'Color', customColor1, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_cornet_s, r48_values_cornet_s, 'o-', 'Color', customColor2, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_alexnet, r48_values_alexnet, 'o-', 'Color', customColor3, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_resnet18, r48_values_resnet18, 'o-', 'Color', customColor4, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_vgg16, r48_values_vgg16, 'o-', 'Color', customColor5, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_gabor, r48_values_gabor, 'o-', 'Color', customColor6, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_CLIPvison, r48_values_CLIPvison, 'o-', 'Color', customColor7, 'LineWidth', 2.0, 'MarkerSize', 2.5);
hold on;
plot(dims_to_check_CLIPtext, r48_values_CLIPtext, 'o-', 'Color', customColor8, 'LineWidth', 2.0, 'MarkerSize', 2.5);


line([dims_to_check_simclr(end) dims_to_check_simclr(end)], [0, r48_values_simclr(dims_to_check_simclr(end))], 'Color', customColor1, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_cornet_s(end) dims_to_check_cornet_s(end)], [0, r48_values_cornet_s(dims_to_check_cornet_s(end))], 'Color', customColor2, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_alexnet(end) dims_to_check_alexnet(end)], [0, r48_values_alexnet(dims_to_check_alexnet(end))], 'Color', customColor3, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_resnet18(end) dims_to_check_resnet18(end)], [0, r48_values_resnet18(dims_to_check_resnet18(end))], 'Color', customColor4, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_vgg16(end) dims_to_check_vgg16(end)], [0, r48_values_vgg16(dims_to_check_vgg16(end))], 'Color', customColor5, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_gabor(end) dims_to_check_gabor(end)], [0, r48_values_gabor(dims_to_check_gabor(end))], 'Color', customColor6, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_CLIPvison(end) dims_to_check_CLIPvison(end)], [0, r48_values_CLIPvison(dims_to_check_CLIPvison(end))], 'Color', customColor7, 'LineStyle', '--', 'LineWidth', 2.0);
line([dims_to_check_CLIPtext(end) dims_to_check_CLIPtext(end)], [0, r48_values_CLIPtext(dims_to_check_CLIPtext(end))], 'Color', customColor8, 'LineStyle', '--', 'LineWidth', 2.0);

legend('SimCLR', 'CORnet\_S', 'AlexNet', 'ResNet18', 'VGG16','Gabor', 'CLIPvison','CLIPtext', 'Location', 'best');


ylabel('Representational similarity score', 'FontSize', 22);
xlabel('Number of Dimensions Retained', 'FontSize', 22);
title('Prediction of measured RSM', 'FontSize', 20);

xlim([1,105])
ylim([0 1])
set (gca,'xtick',(1 : 10 :105));
set (gca,'Ytick',(0 : 0.1 :1));

hax = gca;
set(gca,'FontSize',18) 
hax.Box = 'off';
hax.LineWidth = 1.5; 

exportgraphics(fig, 'figures/r48_values_against_dims_other_dnns.pdf', 'ContentType', 'vector');

close(fig);





