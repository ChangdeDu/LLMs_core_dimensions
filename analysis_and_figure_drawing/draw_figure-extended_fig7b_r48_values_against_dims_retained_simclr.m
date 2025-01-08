
% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load simclr relevant data
data_dir = fullfile(base_dir,'data/DNNs/simclr');
% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));

% load 48 object RDM
load(fullfile(data_dir,'RDM48_triplet.mat'));
%% Load smaller version of words for each image
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');

% Initialize variables for correlation values and dimensions
dims_to_check = 1:size(spose_embedding, 2); % Increasing dimensions
r48_values_simclr = zeros(length(dims_to_check), 1); % Store correlation values

% Loop over dimensions to calculate r48 for increasing dimensionalities
for d = dims_to_check
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


% Plot the r48 values against the dimensions retained
fig = figure('Position',[800 800 500 500],'color','none');

customColor1 = [.2 .6 .4]; 
customColor2 = [.6 .2 .4]; 
customColor3 = [.2 .4 .6]; 
plot(dims_to_check, r48_values_simclr, 'o-', 'Color', customColor1, 'LineWidth', 2.0, 'MarkerSize', 2.5);

line([32 32], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2.0);
legend('SimCLR', 'Location', 'best');


ylabel('Representational similarity score', 'FontSize', 19);
xlabel('Number of Dimensions Retained', 'FontSize', 19);
title('Prediction of measured RSM', 'FontSize', 18);
xlim([1,35])
ylim([0.45 1.0])
set (gca,'xtick',(1 : 5 :35));
set (gca,'Ytick',(0.45 : 0.05 :1.0));

hax = gca;
set(gca,'FontSize',17) 
hax.Box = 'off';
hax.LineWidth = 2.5; 

exportgraphics(fig, 'figures/r48_values_against_dims_simclr.pdf', 'ContentType', 'vector');

close(fig);





