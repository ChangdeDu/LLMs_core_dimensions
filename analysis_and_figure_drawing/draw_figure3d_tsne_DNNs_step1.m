% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% get dimension labels, short labels and colors

load(fullfile(variable_dir,'labels.mat'))

h = fopen(fullfile(variable_dir,'colors.txt')); % get list of colors in hexadecimal format

col = zeros(0,3);
while 1
    l = fgetl(h);
    if l == -1, break, end
    
    col(end+1,:) = reshape(sscanf(l(2:end).','%2x'),3,[]).'/255; % hex2rgb
    
end
fclose(h);

col(1,:) = [];
col([1 2 3],:) = col([2 3 1],:);

% now adapt colors
colors = col([1 20 3 38 9 7 62 57 13 6 24 25 50 48 36 53 46 28 62 18 15 58 2 11 40 45 27 55 36 30 34 31 41 16 27 61 17 36 57 25 63],:); colors(end+1:49,:) = col(8:56-length(colors),:);
colors(46,:) = colors(46,:)-0.2; % medicine related is too bright, needs to be darker

colors = colors([1 2 3 4 6 5 12 8 10 9 13 11 7 15 18 14 16 19 21 17 22 33 17 23 20 27 26 19 24 37 20 28 47 31 39 30 36 43 29 35 38 9 6 25 49 40 42 37 44 25 41 12 20 45 7 41 46 2 23 34 5 33 13 31 40 32],:);

colors([20 28 30 31 41 42 43 45 50 52 53 55 56 58 59 61 62 63 64 65],:) = 1/255*...
    [[146 78 167];
    [143 141 58];
    [255 109 246];
    [71 145 205];
    [0 118 133];
    [204 186 45];
    [0 222 0];
    [222 222 0];
    [100 100 100];
    [40 40 40];
    [126 39 119];
    [177 177 0];
    [50 50 150];
    [120 120 50];
    [250 150 30];
    [40 40 40];
    [220 220 220];
    [90 170 220];
    [140 205 150];
    [40 170 225]];

clear col h l


% model_names_layer = {'feas_CLIP_ViT_L14', 'caption_feas_CLIP_ViT_L14', 'feas_resnet18','feas_vgg16','feas_cornet_s','feas_alexnet','feas_gabor'};
% model_names_spose = {'CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'resnet18','vgg16','cornet_s','alexnet','gabor'};
model_names_layer = {'feas_simclr','feas_vgg16'};
model_names_spose = {'simclr', 'vgg16'};
for model = 1:length(model_names_layer)
    model_name = model_names_layer{model};
    data_dir = fullfile(base_dir,'data/variables');
    embedding = load(fullfile(data_dir,['things1854image_',model_name,'.mat']));
    embedding = embedding.data;
    layer_sim = corrcoef(embedding');
    dissim = 1-layer_sim;

    % First, get 2d MDS solution
    rng(42) % use fixed random number generator
    [Y2,stress] = mdscale(dissim,2,'criterion','metricstress');

    % Next, to visualize how tsne is run, we set clusters according to the
    % strongest dimension in an object

    [~,clustid] = max(embedding,[],2);

    % Then, based on this solution, initialize t-sne solution with multiple
    % perplexities in parallel (multiscale)
    rng(1)
    perplexity1 = 5; perplexity2 = 30;
    D = dissim / max(dissim(:));
    P = 1/2 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5)); % convert distance to affinity matrix using perplexity
    figure
    colormap(colors)
    Ytsne = tsne_p(P,clustid,Y2);
    save_filename = fullfile(data_dir,['things1854image_',model_name,'_tsne.mat']);  
    save(save_filename, 'Ytsne');
    close();
end


for model = 1:length(model_names_spose)
    model_name = model_names_spose{model};
    data_dir = fullfile(base_dir,'data/DNNs',model_name);
    embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
    spose_sim=embedding2sim(embedding);
    dissim = 1-spose_sim;

    % First, get 2d MDS solution
    rng(42) % use fixed random number generator
    [Y2,stress] = mdscale(dissim,2,'criterion','metricstress');

    % Next, to visualize how tsne is run, we set clusters according to the
    % strongest dimension in an object

    embedding(:,1) = embedding(:,1)*0.1;
    [~,clustid] = max(embedding,[],2);

    % Then, based on this solution, initialize t-sne solution with multiple
    % perplexities in parallel (multiscale)
    rng(1)
    perplexity1 = 5; perplexity2 = 30;
    D = dissim / max(dissim(:));
    P = 1/2 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5)); % convert distance to affinity matrix using perplexity
    figure
    colormap(colors)
    Ytsne = tsne_p(P,clustid,Y2);
    save_filename = fullfile(data_dir,'spose_embedding_sorted_merge_tsne.mat');  
    save(save_filename, 'Ytsne');
    close();
end


