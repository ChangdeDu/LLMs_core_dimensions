% run this script from where it is located
base_dir = pwd;

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))
addpath(fullfile(base_dir,'GPT3.5vsGPT4/util'))
%% Load relevant data for humans
data_dir = fullfile(base_dir,'data/Humans');
load(fullfile(data_dir,'RDM48_triplet.mat'));
load(fullfile(data_dir,'RDM48_triplet_splithalf.mat'));
RDM48_triplet_humans = RDM48_triplet;
RDM48_triplet_split1_humans = RDM48_triplet_split1;
RDM48_triplet_split2_humans = RDM48_triplet_split2;

%% Load relevant data for models
model_names = {'ChatGPT-3.5', 'Gemini_Pro_Vision','Llama3.1', 'Qwen2_VL','CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'resnet18','vgg16','swin_large_patch4_window7_224','simclr','cornet_s','alexnet','gabor','gpt2'};
XTick_names = {'ChatGPT-3.5', 'Gemini\_Pro','Llama3.1', 'Qwen2\_VL','CLIPvision', 'CLIPtext','ResNet18', 'VGG16','ViT\_Swin','SimCLR', 'CORnet\_S','AlexNet','Gabor','GPT2'};
% Initialize arrays to store results
rho_values = zeros(length(model_names), 1);
ci_lower = zeros(length(model_names), 1);
ci_upper = zeros(length(model_names), 1);

for model = 1:length(model_names)
    model_name = model_names{model};
    if strcmp(model_name, 'ChatGPT-3.5')
        data_dir = fullfile(base_dir, ['data/LLMs/', model_name]);
    elseif strcmp(model_name, 'Llama3.1')
        data_dir = fullfile(base_dir, ['data/LLMs/', model_name]);
    elseif strcmp(model_name, 'Gemini_Pro_Vision')
        data_dir = fullfile(base_dir, ['data/MLLMs/', model_name]);
    elseif strcmp(model_name, 'Qwen2_VL')
        data_dir = fullfile(base_dir, ['data/MLLMs/', model_name]);
    else
        data_dir = fullfile(base_dir, ['data/DNNs/', model_name]);
    end
    load(fullfile(data_dir, 'RDM48_triplet.mat'));
    load(fullfile(data_dir, 'RDM48_triplet_splithalf.mat'));
    RDM48_triplet_model = RDM48_triplet;
    RDM48_triplet_split1_model = RDM48_triplet_split1;
    RDM48_triplet_split2_model = RDM48_triplet_split2;

    % compute the human consistency  [ref. Rajalingham et al., 2018, JNeurosci]
    r48_mh = corr(squareformq(RDM48_triplet_model),squareformq(RDM48_triplet_humans));
    % run 1000 bootstrap samples for confidence intervals, bootstrap cannot 
    % be done across objects because otherwise it's biased
    rng(2)
    rnd = randi(nchoosek(48,2),nchoosek(48,2),1000);
    c1 = squareformq(RDM48_triplet_model);
    c2 = squareformq(RDM48_triplet_humans);
    for i = 1:1000
        r48_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
    end
    r48_mh_ci95_lower = tanh(atanh(r48_mh) - 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI
    r48_mh_ci95_upper = tanh(atanh(r48_mh) + 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI

    r48_mm = corr(squareformq(RDM48_triplet_split1_model),squareformq(RDM48_triplet_split2_model));
    r48_hh = corr(squareformq(RDM48_triplet_split1_humans),squareformq(RDM48_triplet_split2_humans));

    numerator = r48_mh;
    denominator = sqrt(r48_mm * r48_hh);
    rho = numerator / denominator
    
    % Store results
    rho_values(model) = rho;
    ci_lower(model) = r48_mh_ci95_lower + (rho-r48_mh);
    ci_upper(model) = r48_mh_ci95_upper + (rho-r48_mh);
    
    % add human-human consistency
    %%%%%%%%%%%%%%%
    c1_hh = squareformq(RDM48_triplet_split1_humans);
    c2_hh = squareformq(RDM48_triplet_split2_humans);
    for i = 1:1000
        r48_boot_hh(:,i) = corr(c1_hh(rnd(:,i)),c2_hh(rnd(:,i)));
    end
    r48_hh_ci95_lower = tanh(atanh(r48_hh) - 1.96*std(atanh(r48_boot_hh),[],2)); % reflects 95% CI
    r48_hh_ci95_upper = tanh(atanh(r48_hh) + 1.96*std(atanh(r48_boot_hh),[],2)); % reflects 95% CI
    %%%%%%%%%%%%%
end
rho_values = [r48_hh; rho_values];
ci_lower = [r48_hh_ci95_lower; ci_lower];
ci_upper = [r48_hh_ci95_upper; ci_upper];
model_names = ['Human', model_names];
XTick_names = ['Human', XTick_names];

% Plotting
fig = figure('Position',[1500 1500 1400 1000],'color','none');
colors = [
    [.2 .4 .6];
    [.2 .6 .4];
    [.6 .2 .4]; 
    [.7 .4 .6];
    [.8 .2 .4];
    [.4 .2 .8]; 
    [.8 .5 .3]; 
    [.2 .3 .6];
    [.3 .7 .7]; 
    [.7 .1 .6]; 
    [.2 .4 .4]; 
    [.5 .7 .5]; 
    [.9 .1 .6]; 
    [.5 .1 .6]; 
    [.2 .5 .4]
];
for i = 1:length(rho_values)
    ha(i) = bar(i, rho_values(i), 'FaceColor', colors(i, :), 'EdgeColor', 'none', 'BarWidth', 0.5);
    hold on;
end

e = errorbar(1:length(model_names), rho_values, rho_values - ci_lower, ci_upper - rho_values, 'k', 'LineStyle', 'none');
e.LineWidth = 4;  
set(gca, 'XTickLabel', XTick_names, 'XTick', 1:length(model_names));
xtickangle(45);
ylabel('Human consistency measured by RSA', 'FontSize', 38);
% legend(XTick_names, 'Location', 'best', 'FontSize', 22);
% title('Model Comparisons',  'FontSize', 38);
ymin = floor(min(rho_values)*10)/10; 
ymax = ceil(max(rho_values)*10)/10;  
set(gca, 'YTick', yticks);
% ytickformat('%.1f');
xlim([0.5, length(model_names) + 0.5]);
ylim([0, ymax]);  
% grid on 
hh = hline(0.3:0.2:0.8,'-k');
set(hh,'Color',[1,1,1]*0.8)

hax = gca;
set(gca,'FontSize',31) 
hax.TickDir = 'both';
% hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1.5;
hax.Box = 'off';

exportgraphics(fig, 'figures/human_behavioral_consistency_barplot.pdf', 'ContentType', 'vector');
close(fig);



