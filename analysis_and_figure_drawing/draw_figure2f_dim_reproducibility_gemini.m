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
spose_embedding = load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
dim = size(spose_embedding,2);
load(fullfile(variable_dir,'im.mat'))

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

fig = figure('Position',[800 800 624 500],'color','none');
hold on
x = [1:dim dim:-1:1];
y = [lower_bound' upper_bound(end:-1:1)'];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
plot(mean_reproducibility,'k','linewidth',2.0)
plot(reproducibility,'o','MarkerFaceColor',[.6 .2 .4],'MarkerEdgeColor','none','MarkerSize',3)
ylim([0 1])
xlim([0 dim+1])
set(gca,'FontSize',17) 
ylabel('Reproducibility score', 'FontSize', 21);
xlabel('MLLM embedding dimension number', 'FontSize', 21);
hax = gca;
hax.LineWidth = 1.5; 
if dosave
    exportgraphics(fig, 'dim_reproducibility_gemini.pdf', 'ContentType', 'vector');
end

close(fig);

% Test correlation between rank of reliability and dimension number
[~,reproducibility_ind] = sort(mean_reproducibility,'descend');
r_rank = corr((1:dim)',reproducibility_ind);

% run 100000 permutations
rng(1)
[~,perm] = sort(rand(dim,100000));
r_rank_perm = corr(perm,reproducibility_ind);
% is obviously never exceeded (smaller sign because it's a negative correlation)
p = mean([r_rank_perm;r_rank] >= r_rank);

% run 1000 bootstrap samples for confidence intervals
rng(2)
rnd = randi(dim,dim,1000);
for i = 1:1000
    r_rank_boot(:,i) = corr(rnd(:,i),reproducibility_ind(rnd(:,i)));
end
r_rank_ci95_lower = tanh(atanh(r_rank) - 1.96*std(atanh(r_rank_boot),[],2)); % reflects 95% CI
r_rank_ci95_upper = tanh(atanh(r_rank) + 1.96*std(atanh(r_rank_boot),[],2)); % reflects 95% CI