
% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes

% t-SNE from: https://lvdmaaten.github.io/tsne/#implementations
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
spose_sim=embedding2sim(spose_embedding66);
dissim = 1-spose_sim;

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

%% Load smaller version of images, words, and unique IDs for each image

load(fullfile(variable_dir,'im.mat'))
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'unique_id.mat'))
load(fullfile(variable_dir,'words48.mat'))

% TODO: perhaps still sorting error with im and imwords
% now sort images according to unique_id
[~,i] = sort(unique_id);
[~,j] = sort(imwords);
imwords(i) = imwords(j);
im(i) = im(j);

[~,~,wordposition48] = intersect(words48,words,'stable');


% First, get 2d MDS solution
rng(42) % use fixed random number generator
[Y2,stress] = mdscale(dissim,2,'criterion','metricstress');

% Next, to visualize how tsne is run, we set clusters according to the
% strongest dimension in an object
spose_embedding = spose_embedding66;
spose_embedding(:,1) = spose_embedding(:,1)*0.1;
[~,clustid] = max(spose_embedding,[],2);

% Then, based on this solution, initialize t-sne solution with multiple
% perplexities in parallel (multiscale)
rng(1)
perplexity1 = 5; perplexity2 = 30;
D = dissim / max(dissim(:));
P = 1/2 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5)); % convert distance to affinity matrix using perplexity
figure
colormap(colors)
Ytsne = tsne_p(P,clustid,Y2);

