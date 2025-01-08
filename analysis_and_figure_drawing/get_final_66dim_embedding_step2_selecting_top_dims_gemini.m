clear all;
% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data
% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
dim = size(spose_embedding,2);
% get dot product (i.e. proximity)
dot_product49 = spose_embedding*spose_embedding';
dosave = 1;

load(fullfile(variable_dir,'sortind.mat')); % need this because original order is wrong
refdir = fullfile(data_dir,'reference_models_gemini_spose');
for i_model = 1:20
    fn = dir(fullfile(refdir,sprintf('s%02i',i_model),'*.txt'));
    fn = fullfile(fn(end).folder,fn(end).name);
    tmp = load(fn);
    % remove empty dimensions
    tmp2 = tmp(:,any(tmp>0.1));
    reference_models{i_model,1} = tmp2(sortind,:);
    n_dim_reference(i_model) = size(reference_models{i_model},2);
end

% Correlate dimensions (this slightly overestimates the performance, given
% that each dimension can be picked several times, but there is no other
% way - otherwise some dimensions would go unmatched)
for i_model = 1:20
    reproducibility(:,i_model) = max(corr(spose_embedding,reference_models{i_model}),[],2);
end

% test split-half prediction
for i_model = 1:20
    [~,maxind(:,i_model)] = max(corr(spose_embedding(1:2:end,:),reference_models{i_model}(1:2:end,:)),[],2);
    [~,maxind2(:,i_model)] = max(corr(spose_embedding(2:2:end,:),reference_models{i_model}(2:2:end,:)),[],2);
    c1 = corr(spose_embedding(1:2:end,:),reference_models{i_model}(1:2:end,:));
    c2 = corr(spose_embedding(2:2:end,:),reference_models{i_model}(2:2:end,:));
    for i = 1:dim, tmp1(i,i_model) = c1(i,maxind2(i,i_model)); tmp2(i,i_model) = c2(i,maxind(i,i_model)); end
end

% fisher-z convert before averaging across models
mean_reproducibility = mean(atanh(reproducibility),2);
reproducibility_ci95 = 1.96*std(atanh(reproducibility),[],2)/sqrt(20);

% for plotting, the upper bound will be mean + 95% CI, then conversion back
% to correlation, same for lower bound
upper_bound = tanh(mean_reproducibility+reproducibility_ci95);
lower_bound = tanh(mean_reproducibility-reproducibility_ci95);
% now update mean reproducibility, as well
mean_reproducibility = tanh(mean_reproducibility);

% choose top dimensions
top_num = 66;
[sorted_values, sorted_indices] = sort(mean_reproducibility, 'descend');
top_indices = sorted_indices(1:top_num);
spose_embedding_topd = spose_embedding(:,top_indices);
column_sums = sum(spose_embedding_topd);
[~, sorted_indices] = sort(column_sums, 'descend');
spose_embedding_66d_sorted = spose_embedding_topd(:, sorted_indices);

filename='data/MLLMs/Gemini_Pro_Vision/spose_embedding_66d_sorted_gemini.txt';
save(filename,'spose_embedding_66d_sorted', '-ascii')
