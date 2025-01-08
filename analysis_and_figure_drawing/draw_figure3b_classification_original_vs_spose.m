% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

model_names_layer = {'feas_CLIP_ViT_L14', 'caption_feas_CLIP_ViT_L14', 'feas_simclr','feas_resnet18','feas_vgg16','feas_cornet_s','feas_alexnet','feas_gabor'};
model_names_spose = {'CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'simclr','resnet18','vgg16','cornet_s','alexnet','gabor'};
XTick_names = {'CLIPvision', 'CLIPtext','SimCLR','ResNet18', 'VGG16', 'CORnet\_S','AlexNet','Gabor'};

behav_predict_acc_layer = zeros(length(model_names_layer), 1); % Store acc values
behav_predict_chance_level_layer= zeros(length(model_names_layer), 1); 
behav_predict_lower_bound_layer = zeros(length(model_names_layer), 1); 
behav_predict_upper_bound_layer= zeros(length(model_names_layer), 1); 

behav_predict_acc_spose = zeros(length(model_names_spose), 1); % Store acc values
behav_predict_chance_level_spose= zeros(length(model_names_spose), 1); 
behav_predict_lower_bound_spose = zeros(length(model_names_spose), 1); 
behav_predict_upper_bound_spose= zeros(length(model_names_spose), 1); 

for model = 1:length(model_names_layer)
    model_name = model_names_layer{model};
    layer_embedding = ['things1854image_',model_name,'.mat'];
    data_dir = fullfile(base_dir,'data/variables');
    [acc, predlabel, catlabels] = predict_category_for_layer_fea(layer_embedding, base_dir, data_dir,variable_dir);
    [chance_level, lower_bound, upper_bound] = compute_chance_level(predlabel, catlabels);

    behav_predict_acc_layer(model) = acc;
    behav_predict_chance_level_layer(model) = chance_level;
    behav_predict_lower_bound_layer(model) = lower_bound;
    behav_predict_upper_bound_layer(model) = upper_bound;
end


for model = 1:length(model_names_spose)
    model_name = model_names_spose{model};
    embedding = 'spose_embedding_sorted_merge.txt';
    data_dir = fullfile(base_dir,'data/DNNs',model_name);
    [acc, predlabel, catlabels] = predict_category(embedding, base_dir, data_dir,variable_dir);
    [chance_level, lower_bound, upper_bound] = compute_chance_level(predlabel, catlabels);

    behav_predict_acc_spose(model) = acc;
    behav_predict_chance_level_spose(model) = chance_level;
    behav_predict_lower_bound_spose(model) = lower_bound;
    behav_predict_upper_bound_spose(model) = upper_bound;
end

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%

fig = figure('Position',[800 800 500 500],'color','none');

customColor1 = [.2 .2 .9]; 
customColor2 = [.9 .2 .2]; 

plot(1:length(model_names_layer), behav_predict_acc_layer, 'o-', 'MarkerFaceColor', customColor1, 'Color', customColor1, 'LineWidth', 3, 'MarkerSize', 3.5);
hold on;
plot(1:length(model_names_spose), behav_predict_acc_spose, 's-', 'MarkerFaceColor', customColor2, 'Color', customColor2, 'LineWidth', 3, 'MarkerSize', 3.5);
legend('Original', 'SPoSE', 'Location', 'best');

xticks(1:length(XTick_names));
xticklabels(XTick_names);

ylabel('Categorization performance (%)', 'FontSize', 19);

ylim([10 100])
set (gca,'Ytick',(10 : 10 :100));
xtickangle(45);
hax = gca;
set(gca,'FontSize',17) 
hax.Box = 'off';
hax.LineWidth = 1.5; 

exportgraphics(fig, 'figures/classification_acc_layer_vs_spose.pdf', 'ContentType', 'vector');

close(fig);


