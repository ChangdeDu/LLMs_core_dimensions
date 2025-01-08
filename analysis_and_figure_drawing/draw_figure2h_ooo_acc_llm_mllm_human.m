% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))


%% ChatGPT-3.5: Predict behavior and similarity
%% Load relevant data
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile(data_dir,'triplet_dataset/validationset_ChatGPT_3.5.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata(triplet_testdata==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata = triplet_testdata-10000;

dosave = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product66(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_llm = 100*mean(behav_predict==1);

num_permutations = 1000;
permuted_accuracies_llm = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_llm(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_llm = mean(permuted_accuracies_llm);
% chance level 95% CI
lower_bound_llm = prctile(permuted_accuracies_llm, 2.5);
upper_bound_llm = prctile(permuted_accuracies_llm, 97.5);
fprintf('Chance Level LLM: %.4f\n', chance_level_llm);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_llm, upper_bound_llm);


% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_llm = 1.96*std(behav_predict_obj)/sqrt(1854);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
h = fopen(fullfile(data_dir,'triplets_noiseceiling_ChatGPT_table.csv'),'r');
NCdat = zeros(20000,4);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000)

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
fig = figure('Position',[800 800 600 500],'color','none');
% first plot noise ceiling
x =  [2 6 6 2];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on


%% Gemini_Pro_Vision: Predict behavior and similarity
%% Load relevant data
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile(data_dir,'triplet_dataset/validationset_Gemini_Pro_Vision.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata(triplet_testdata==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata = triplet_testdata-10000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product66(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_mllm = 100*mean(behav_predict==1);

num_permutations = 1000;
permuted_accuracies_mllm = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_mllm(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_mllm = mean(permuted_accuracies_mllm);
% chance level 95% CI
lower_bound_mllm = prctile(permuted_accuracies_mllm, 2.5);
upper_bound_mllm = prctile(permuted_accuracies_mllm, 97.5);
fprintf('Chance Level MLLM: %.4f\n', chance_level_mllm);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_mllm, upper_bound_mllm);


% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_MLLM = 1.96*std(behav_predict_obj)/sqrt(1854);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
h = fopen(fullfile(data_dir,'triplets_noiseceiling_Gemini_Vision_table.csv'),'r');
NCdat = zeros(20000,4);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000)

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
x = [8 12 12 8];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on

%% Humans: Predict behavior and similarity
%% Load relevant data
data_dir = fullfile(base_dir,'data/Humans');
% load embedding
spose_embedding66 = load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding66*spose_embedding66';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile(data_dir,'triplet_dataset/validationset.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it
load(fullfile(variable_dir,'sortind.mat'));
for i_obj = 1:1854
    triplet_testdata(triplet_testdata==sortind(i_obj)) = 10000+i_obj;
end
triplet_testdata = triplet_testdata-10000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product66(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product66(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc = 100*mean(behav_predict==1);

num_permutations = 1000;
permuted_accuracies_human = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_human(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_human = mean(permuted_accuracies_human);
% chance level 95% CI
lower_bound_human = prctile(permuted_accuracies_human, 2.5);
upper_bound_human = prctile(permuted_accuracies_human, 97.5);
fprintf('Chance Level Human: %.4f\n', chance_level_human);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_human, upper_bound_human);


% get prediction for each object
for i_obj = 1:1854
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95 = 1.96*std(behav_predict_obj)/sqrt(1854);


%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%

h = fopen(fullfile(data_dir,'triplets_noiseceiling.csv'),'r');
NCdat = zeros(20000,5);
cnt = 0;
while 1
    l = fgetl(h);
    if l == -1, break, end
    l2 = strsplit(l);
    cnt = cnt+1;
    NCdat(cnt,:) = str2double(l2);
end
fclose(h);

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);

% get number of triplets for each
for i = 1:1000
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:1000
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(1000)

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%

% first plot noise ceiling
x = [14 18 18 14];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
hold on

% now plot results
ha3 = bar(4,behav_predict_acc_llm,'FaceColor',[.2 .6 .4],'EdgeColor','none','BarWidth',4);
hb3 = errorbar(4, behav_predict_acc_llm, behav_predict_acc_ci95_llm,'Color',[0 0 0],'LineWidth',4);

ha4 = bar(10,behav_predict_acc_mllm,'FaceColor',[.6 .2 .4],'EdgeColor','none','BarWidth',4);
hb4 = errorbar(10, behav_predict_acc_mllm, behav_predict_acc_ci95_MLLM,'Color',[0 0 0],'LineWidth',4);

ha5 = bar(16, behav_predict_acc,'FaceColor',[.2 .4 .6],'EdgeColor','none','BarWidth',4);
hb5 = errorbar(16, behav_predict_acc, behav_predict_acc_ci95,'Color',[0 0 0],'LineWidth',4);

hb = plot([2,6],[chance_level_llm chance_level_llm],'-r','LineWidth',2);
hb3 = errorbar(4, chance_level_llm, (upper_bound_llm -lower_bound_llm)/2,'Color',[0 0 0],'LineWidth',2);
hc = plot([8,12],[chance_level_mllm chance_level_mllm],'-r','LineWidth',2);
hb5 = errorbar(10, chance_level_mllm, (upper_bound_mllm-lower_bound_mllm)/2,'Color',[0 0 0],'LineWidth',2);
hd = plot([14,18],[chance_level_human chance_level_human],'-r','LineWidth',2);
hb4 = errorbar(16, chance_level_human, (upper_bound_human-lower_bound_human)/2,'Color',[0 0 0],'LineWidth',2);


% text(1.6, 63.5, 'LLM noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 13);
% text(7, 69.3, 'MLLM noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 13);
text(14, 69.3, ' noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
text(14.0, 35, 'chance', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'red', 'FontSize', 16);
text(4, 47, 'ChatGPT-3.5', 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize', 17);
text(10, 47, 'Gemini Pro Vision', 'Rotation',90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize',  17);
text(16, 47, 'Human', 'Rotation',90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white','FontSize',  17);
axis equal
xlim([-1,20])
ylim([30 76])

% 添加纵向标题
ylabel('Odd-one-out accuracy (%)', 'FontSize', 18);
ylim([30 80])
set (gca,'Ytick',(30 : 5 :80));

hax = gca;
set(gca,'FontSize',17) % 字号可以根据需要进行调整
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1.5;
hax.Box = 'off';

if dosave
    exportgraphics(fig, 'figures/noisecelling_llm_mllm_human.pdf', 'ContentType', 'vector');
end
close(fig);
