
% run this script from where it is located
base_dir = pwd;

variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data

% load embedding
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
spose_embedding_chatgpt = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
spose_embedding_chatgpt(:,1) = spose_embedding_chatgpt(:,1)*0.1; % The first dimension is scaled to avoid affecting the visualization of the other dimensions
labels_short_chatgpt = importdata(fullfile(variable_dir,'labels_short_66_chatgpt.txt'));

% load embedding
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
spose_embedding_gemini = load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
spose_embedding_gemini(:,1) = spose_embedding_gemini(:,1)*0.1; % The first dimension is scaled to avoid affecting the visualization of the other dimensions
labels_short_gemini = importdata(fullfile(variable_dir,'labels_short_66_gemini.txt'));

% load embedding
data_dir = fullfile(base_dir,'data/Humans/');
spose_embedding_human = load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));
labels_short_human = importdata(fullfile(variable_dir,'labels_short.txt'));

% load answers from participants labeling the images
load(fullfile(variable_dir,'im.mat'))
load(fullfile(variable_dir,'words.mat'))


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

scaling = 2.8;
tempFolder = 'temp_figures_chatgpt_gemini_human_rose';
if ~exist(tempFolder, 'dir')
    mkdir(tempFolder);
end

for i_example = [41 166 422 591 714 726 848 1292 1658] %1:1854
    v0 = [];
    f0 = [];
    v1 = [];
    ct = 0;
    [x0,y0] = pol2cart(repmat(linspace(0,2*pi,66+1),[66 1]),scaling*repmat(spose_embedding_chatgpt(i_example,1:66)',[1 66+1]));
    [th,r] = cart2pol(x0,y0); [x1,y1] = pol2cart(th,r-0.05);
    for i = 1:66
        ct = ct+1;
        v0 = [v0; [0 0; x0(i,i) y0(i,i); x0(i,i+1) y0(i,i+1)]];
        f0 = [f0; ((ct-1)*3 + (1:3))];
        v1 = [v1; [0 0; x1(i,i) y1(i,i); x1(i,i+1) y1(i,i+1)]];
    end
   
    
    v00 = [];
    f00 = [];
    v2 = [];
    ct = 0;
    [x0,y0] = pol2cart(repmat(linspace(0,2*pi,66+1),[66 1]),scaling*repmat(spose_embedding_gemini(i_example,1:66)',[1 66+1]));
    [th00,r00] = cart2pol(x0,y0); [x1,y1] = pol2cart(th00,r00-0.05);
    for i = 1:66
        ct = ct+1;
        v00 = [v00; [0 0; x0(i,i) y0(i,i); x0(i,i+1) y0(i,i+1)]];
        f00 = [f00; ((ct-1)*3 + (1:3))];
        v2 = [v2; [0 0; x1(i,i) y1(i,i); x1(i,i+1) y1(i,i+1)]];
    end


    v000 = [];
    f000 = [];
    v3 = [];
    ct = 0;
    [x0,y0] = pol2cart(repmat(linspace(0,2*pi,66+1),[66 1]),scaling*repmat(spose_embedding_human(i_example,1:66)',[1 66+1]));
    [th000,r000] = cart2pol(x0,y0); [x1,y1] = pol2cart(th000,r000-0.05);
    for i = 1:66
        ct = ct+1;
        v000 = [v000; [0 0; x0(i,i) y0(i,i); x0(i,i+1) y0(i,i+1)]];
        f000 = [f000; ((ct-1)*3 + (1:3))];
        v3 = [v3; [0 0; x1(i,i) y1(i,i); x1(i,i+1) y1(i,i+1)]];
    end


    fig = figure('Position',[870 2043 2076 510]);

    subtightplot(1,4,1)
    currfn = ['data\THINGS_visual_stimuli_1854\image_', num2str(i_example), '_ori.jpg'];
    img = imread(currfn); 
    img = imresize(img, [412, 412]);
    newImg = uint8(255*ones(size(img, 1) + 292, size(img, 2) + 292, size(img, 3)));
    newImg(147:end-146, 147:end-146, :) = img;
    newImg = newImg(1:618,1:618,:);
    newImg = imresize(newImg, [128, 128]);
    imagesc(newImg);
    a = gca;
    a.XTick = [];
    a.YTick = [];
    axis off 
    maxWidth = 30;  
    description = words{i_example};
    result = lower(description);  
    result(1) = upper(result(1)); 
    textArray = textwrap({result}, maxWidth);
    wrappedText = strjoin(textArray, '\n');
    title(sprintf(wrappedText), 'FontSize', 30, 'Position', [size(newImg, 2)/2+10, 27]);


    subtightplot(1,4,2)
    patch('faces',f0,'vertices',v0,'FaceVertexCData',linspace(0,1,66)','FaceColor','flat','edgecolor','none','facealpha',0.5)
    colormap(colors)
    axis off square equal tight

    hold on
    clear ht rd
    rot = linspace(0,2*pi,66+1);
    rot = conv(rot,[0.5 0.5]);
    rot = rot(2:end-1);
    for i = 1:66
        if r(i,1)<2.0, continue, end
        vind = 3*(i-1);
        ht(i) = text(mean(v1(vind+(2:3),1)),mean(v1(vind+(2:3),2)), labels_short_chatgpt{i}, 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        ht(i).Rotation = rad2deg(rot(i));
        rd(i) = mod(rad2deg(rot(i)),180);
        ht(i).FontSize = 28; % was 18
        ht(i).FontName = 'Avenir Next';
        ht(i).HorizontalAlignment = 'right';
        if ht(i).Rotation>90 && ht(i).Rotation<270
            ht(i).Rotation = ht(i).Rotation+180;
            ht(i).HorizontalAlignment = 'left';
        end
    end


    subtightplot(1,4,3)
    patch('faces',f00,'vertices',v00,'FaceVertexCData',linspace(0,1,66)','FaceColor','flat','edgecolor','none','facealpha',0.5)
    colormap(colors)
    axis off square equal tight

    hold on
    clear ht rd
    rot = linspace(0,2*pi,66+1);
    rot = conv(rot,[0.5 0.5]);
    rot = rot(2:end-1);
    for i = 1:66
        if r00(i,1)<2.0, continue, end
        vind = 3*(i-1);
        ht(i) = text(mean(v2(vind+(2:3),1)),mean(v2(vind+(2:3),2)), labels_short_gemini{i}, 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        ht(i).Rotation = rad2deg(rot(i));
        rd(i) = mod(rad2deg(rot(i)),180);
        ht(i).FontSize = 28; % was 18
        ht(i).FontName = 'Avenir Next';
        ht(i).HorizontalAlignment = 'right';
        if ht(i).Rotation>90 && ht(i).Rotation<270
            ht(i).Rotation = ht(i).Rotation+180;
            ht(i).HorizontalAlignment = 'left';
        end
    end


    subtightplot(1,4,4)
    patch('faces',f000,'vertices',v000,'FaceVertexCData',linspace(0,1,66)','FaceColor','flat','edgecolor','none','facealpha',0.5)
    colormap(colors)
    axis off square equal tight

    hold on
    clear ht rd
    rot = linspace(0,2*pi,66+1);
    rot = conv(rot,[0.5 0.5]);
    rot = rot(2:end-1);
    for i = 1:66
        if r000(i,1)<2.0, continue, end
        vind = 3*(i-1);
        ht(i) = text(mean(v3(vind+(2:3),1)),mean(v3(vind+(2:3),2)), labels_short_human{i}, 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        ht(i).Rotation = rad2deg(rot(i));
        rd(i) = mod(rad2deg(rot(i)),180);
        ht(i).FontSize = 28; % was 18
        ht(i).FontName = 'Avenir Next';
        ht(i).HorizontalAlignment = 'right';
        if ht(i).Rotation>90 && ht(i).Rotation<270
            ht(i).Rotation = ht(i).Rotation+180;
            ht(i).HorizontalAlignment = 'left';
        end
    end

    figFile = fullfile(tempFolder, sprintf('rose_%d_chatgpt_gemini_human.pdf', i_example));
    exportgraphics(fig, figFile, 'ContentType', 'vector');
    close(fig);
    
end
