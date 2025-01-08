
% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data

% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';
% load similarity computed from embedding (using embedding2sim.m)
load(fullfile(data_dir,'chatgpt_spose_similarity.mat'))
% spose_sim=embedding2sim(spose_embedding66);
dissim = 1-spose_sim;
% load test set
triplet_testdata66 = load(fullfile(data_dir,'triplet_dataset\validationset_ChatGPT_3.5.txt'))+1; % 0 index -> 1 index



%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata66(triplet_testdata66==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata66 = triplet_testdata66-10000;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata66),1);
behav_predict_prob = zeros(length(triplet_testdata66),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata66)
    sim(1) = dot_product66(triplet_testdata66(i,1),triplet_testdata66(i,2));
    sim(2) = dot_product66(triplet_testdata66(i,1),triplet_testdata66(i,3));
    sim(3) = dot_product66(triplet_testdata66(i,2),triplet_testdata66(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata66==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata66==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95 = 1.96*std(behav_predict_obj)/sqrt(1854);

%% Calculate how the behavioral prediction changes when eliminating dimensions with small weight

dosave = 1;

% do this for each obj separately, i.e. get index for each object across 49
% dimensions, then eliminate one by one

fn1 = fullfile(data_dir,'chatgpt_spose_similarity_reduced.mat');
fn2 = fullfile(data_dir,'chatgpt_spose_embedding66_reduced.mat');

if ~exist(fn1,'file') || ~exist(fn2','file')
    [~,embedding_sortind] = sort(spose_embedding66,2);
    disp('Getting reduced versions of embeddings and converting them to similarity.')
    disp('This takes about 10-15min on a regular laptop but only needs to be run once.')
    for i_dim = 1:66
        fprintf('.')
        % make 66 reduced versions, make 66 reduced similarity matrices
        if i_dim == 1
            chatgpt_spose_embedding66_reduced{i_dim,1} = spose_embedding66;
        else
            chatgpt_spose_embedding66_reduced{i_dim,1} = chatgpt_spose_embedding66_reduced{i_dim-1,1};
        end
        for i = 1:1854
            chatgpt_spose_embedding66_reduced{i_dim,1}(i,embedding_sortind(i,i_dim)) = 0;
        end
        chatgpt_spose_similarity_reduced{i_dim,1} = embedding2sim(chatgpt_spose_embedding66_reduced{i_dim,1});
        
    end
    fprintf('\n')
    save(fn1,'chatgpt_spose_similarity_reduced')
    save(fn2,'chatgpt_spose_embedding66_reduced')
else
    load(fn1)
    load(fn2)
end

clear sim
for i_dim = 1:66
    rng(42) % for reproducibility
    behav_predict = zeros(length(triplet_testdata66),1);
    dot_product66_reduc = chatgpt_spose_embedding66_reduced{i_dim}*chatgpt_spose_embedding66_reduced{i_dim}';
    for i = 1:length(triplet_testdata66)
        sim(1) = dot_product66_reduc(triplet_testdata66(i,1),triplet_testdata66(i,2));
        sim(2) = dot_product66_reduc(triplet_testdata66(i,1),triplet_testdata66(i,3));
        sim(3) = dot_product66_reduc(triplet_testdata66(i,2),triplet_testdata66(i,3));
        [m,mi] = max(sim);
        if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
        behav_predict(i,1) = mi;
    end
    % get overall prediction (predict choice == 1)
    behav_predict_acc_reduc(i_dim) = 100*mean(behav_predict==1);
    % get prediction for each object
    for i_obj = 1:1854
        behav_predict_obj_reduc(i_obj,i_dim) = 100*mean(behav_predict(any(triplet_testdata66==i_obj,2))==1);
    end
    % get standard error for this value across objects
    behav_predict_acc_reduc_ci95(i_dim,1) = 1.96* std(behav_predict_obj_reduc(:,i_dim))/sqrt(1854);
end

% now reverse it all
behav_predict_acc_reduc = behav_predict_acc_reduc(end:-1:1);
behav_predict_obj_reduc = behav_predict_obj_reduc(:,end:-1:1);
behav_predict_acc_reduc_ci95 = behav_predict_acc_reduc_ci95(end:-1:1);

cutoff95 = (0.95*behav_predict_acc-100/3)+100/3;
cutoff99 = (0.99*behav_predict_acc-100/3)+100/3;

mindim = find(behav_predict_acc_reduc>cutoff95,1,'first')-1; % -1 because we need to start counting at 0
maxdim = find(behav_predict_acc_reduc<cutoff99,1,'last')-1;
fprintf('We need between %i and %i dimensions to reach 95-99%% performance in predicting individual trials.\n',mindim,maxdim)

% compare similarity matrices
for i_dim = 1:66
    r_reduc(i_dim) = corr(squareformq(spose_sim),squareformq(chatgpt_spose_similarity_reduced{i_dim}));
end

% reverse r_reduc
r_reduc = r_reduc(end:-1:1);

mindim2 = find(r_reduc.^2>0.95,1,'first')-1; % -1 because we need to start counting at 0
maxdim2 = find(r_reduc.^2<0.99,1,'last')-1;
fprintf('We need between %i and %i dimensions to explain 95-99%% variance in similarity.\n',mindim2,maxdim2)


fig = figure('Position',[1000 1000 400 400],'color','none');
cutoff95 = (0.95*behav_predict_acc-100/3)+100/3;
cutoff99 = (0.99*behav_predict_acc-100/3)+100/3;
% also, add noise ceiling
hold on
x = [0 67 67 0];
y = [cutoff95 cutoff95 cutoff99 cutoff99];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
% hc.FaceAlpha = 0.3;
x = [mindim maxdim maxdim mindim];
y = [70 70 0 0];
hcb = patch(x,y,[.2 .6 .4]);
hcb.EdgeColor = 'none';
% hcb.FaceAlpha = 0.5;
plot(0:66,[behav_predict_acc_reduc behav_predict_acc],'k','LineWidth',3)
plot([0 66],[100/3 100/3],'--r')
text(50, 35, 'chance', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
xlim([0 66])
ylim([0 70])
xlabel('Number of dimensions retained', 'FontSize', 16)
ylabel('Accuracy (%)', 'FontSize', 16)
title('Prediction of LLM behavior', 'FontSize', 16)

ax = gca;
set(ax, 'FontSize', 10);

if dosave
    exportgraphics(fig, 'acc_vs_dims_retained.pdf', 'ContentType', 'vector');
end

close(fig);

fig = figure('Position',[1000 1000 400 400],'color','none');
% 95% means here means 95% and 99% means 99%
hold on
x = [0 67 67 0];
y = [95 95 99 99];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
% hc.FaceAlpha = 0.3;
x = [mindim2 maxdim2 maxdim2 mindim2];
y = [108.2 108.2 0 0];
hcb = patch(x,y,[.2 .6 .4]);
hcb.EdgeColor = 'none';
% hcb.FaceAlpha = 0.5;
xlim([0 66])
ylim([0 108.2])
plot(0:66,[100*r_reduc.^2 100],'k','LineWidth',3)
xlabel('Number of dimensions retained', 'FontSize', 16)
ylabel('Variance explained (%)', 'FontSize', 16)
title('Relationship to full similarity matrix', 'FontSize', 16)

if dosave
    exportgraphics(fig, 'var_vs_dims_retained.pdf', 'ContentType', 'vector');
end

close(fig);
