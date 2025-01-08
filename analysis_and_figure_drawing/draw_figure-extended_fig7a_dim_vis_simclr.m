
% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data/DNNs/simclr');
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data
% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));

%% get dimension labels, short labels and colors
labels_short = importdata(fullfile(variable_dir,'labels_short_32_simclr.txt'));
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


%% Show examples
dosave = 1;
% sort each dimension
[~,dimsortind] = sort(spose_embedding,'descend');

n_im = 6;
tempFolder = 'temp_figures_simclr';
if ~exist(tempFolder, 'dir')
    mkdir(tempFolder);
end

num_dim = size(spose_embedding,2);
for i_dim = 1:num_dim % chosen dimensions

    fig = figure('Position',[870 2043 2476 310],'color','none');

    for i = 1:n_im

        subtightplot(1,9,i,0.005)
        currfn = ['data\THINGS_visual_stimuli_1854\image_', num2str(dimsortind(i,i_dim)), '_ori.jpg'];
        dimsortind(i,i_dim)
        img = imread(currfn); 
        img = imresize(img, [80, 80]);
        newImg = uint8(255*ones(size(img, 1) + 8, size(img, 2) + 8, size(img, 3)));
        newImg(9:end, 5:end-4, :) = img;
        newImg = newImg(8:88,4:84,:);
        imagesc(newImg);
        if i==4
            title(sprintf('Dimension %i: %s',i_dim,labels_short{i_dim}), 'FontSize', 40)
        end
        axis off square
    end
    
    figFile = fullfile(tempFolder, sprintf('figure_%d_simclr.pdf', i_dim));
    exportgraphics(fig, figFile, 'ContentType', 'vector');
    close(fig);
    
end
