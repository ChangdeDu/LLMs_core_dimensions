% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))


model_names_spose = {'CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'simclr','resnet18','vgg16','cornet_s','alexnet','gabor'};
XTick_names = {'CLIPvision', 'CLIPtext','SimCLR','ResNet18', 'VGG16', 'CORnet\_S','AlexNet','Gabor'};

behav_predict_acc_spose = zeros(length(model_names_spose), 1); % Store acc values
behav_predict_acc_ci95_spose = zeros(length(model_names_spose), 1); % Store acc 95% c.i. values

for model = 1:length(model_names_spose)
    model_name = model_names_spose{model};
    data_dir = fullfile(base_dir,'data/DNNs',model_name);
    % load embedding
    spose_embedding = load(fullfile(data_dir,'spose_embedding_sorted_merge.txt'));
    % get dot product (i.e. proximity)
    dot_product = spose_embedding*spose_embedding';
    
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
        sim(1) = dot_product(triplet_testdata(i,1),triplet_testdata(i,2));
        sim(2) = dot_product(triplet_testdata(i,1),triplet_testdata(i,3));
        sim(3) = dot_product(triplet_testdata(i,2),triplet_testdata(i,3));
        [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
        if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
        behav_predict(i,1) = mi;
        behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
    end
    % get overall prediction (predict choice == 1)
    behav_predict_acc_spose(model) = 100*mean(behav_predict==1);
    
    
    % get prediction for each object
    for i_obj = 1:1854
        behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
        % this below gives us the predictability of each object on average
        % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
        behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
    end
    % get 95% CI for this value across objects
    behav_predict_acc_ci95_spose(model) = 1.96*std(behav_predict_obj)/sqrt(1854);
end


%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%

fig = figure('Position',[800 800 500 500],'color','none');

customColor1 = [.2 .2 .9]; 
customColor2 = [.9 .2 .2]; 

plot(1:length(model_names_spose), behav_predict_acc_spose./behav_predict_acc_layer, 's-', 'MarkerFaceColor', customColor2, 'Color', customColor2, 'LineWidth', 3, 'MarkerSize', 3.5);


xticks(1:length(XTick_names));
xticklabels(XTick_names);

ylabel('Accuracy ratio', 'FontSize', 19);
ylim([0.3 1])
set (gca,'Ytick',(0.3 : 0.1 :1));

xtickangle(45);
hax = gca;
set(gca,'FontSize',17) 
hax.Box = 'off';
hax.LineWidth = 1.5; 

exportgraphics(fig, 'figures/prediction_acc_layer_vs_spose.pdf', 'ContentType', 'vector');

close(fig);



