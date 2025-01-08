% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))
[~,~,wordposition48] = intersect(words48,words,'stable');
%% Load relevant data for chatgpt3.5
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
load(fullfile(data_dir,'RSM1854_get_from_full_sampling_of_48_objects_ChatGPT-3.5.mat'));
load(fullfile(data_dir,'RSM1854_get_from_full_sampling_of_48_objects_ChatGPT-3.5_splithalf.mat'));
RSM48_triplet = RSM1854_triplet(wordposition48,wordposition48);
RSM48_triplet_split1 = RSM1854_triplet_split1(wordposition48,wordposition48);
RSM48_triplet_split2 = RSM1854_triplet_split2(wordposition48,wordposition48);
RDM48_triplet = 1- RSM48_triplet;
RDM48_triplet_split1 = 1- RSM48_triplet_split1;
RDM48_triplet_split2 = 1- RSM48_triplet_split2;
save_filename = [data_dir,'/RDM48_triplet.mat'];  
save(save_filename, 'RDM48_triplet');
save_filename = [data_dir,'/RDM48_triplet_splithalf.mat'];  
save(save_filename, 'RDM48_triplet_split1', 'RDM48_triplet_split2');

%% Load relevant data for gemini_pro_vision
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
load(fullfile(data_dir,'RSM1854_get_from_full_sampling_of_48_objects_Gemini_Pro_Vision.mat'));
load(fullfile(data_dir,'RSM1854_get_from_full_sampling_of_48_objects_Gemini_Pro_Vision_splithalf.mat'));
RSM48_triplet = RSM1854_triplet(wordposition48,wordposition48);
RSM48_triplet_split1 = RSM1854_triplet_split1(wordposition48,wordposition48);
RSM48_triplet_split2 = RSM1854_triplet_split2(wordposition48,wordposition48);
RDM48_triplet = 1- RSM48_triplet;
RDM48_triplet_split1 = 1- RSM48_triplet_split1;
RDM48_triplet_split2 = 1- RSM48_triplet_split2;
save_filename = [data_dir,'/RDM48_triplet.mat'];  
save(save_filename, 'RDM48_triplet');
save_filename = [data_dir,'/RDM48_triplet_splithalf.mat'];  
save(save_filename, 'RDM48_triplet_split1', 'RDM48_triplet_split2');

%% Load relevant data for other DNNs
model_names = {'CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'gabor', 'vit_base_patch16_224','resnet50','vgg16','resnet18','cornet_s','alexnet','gpt2',  'vit_large_patch16_224','swin_large_patch4_window7_224','simclr','densenet121'};
for model = 1:length(model_names)
    model_name = model_names{model};
    data_dir = fullfile(base_dir,['data/DNNs/',model_name]);
    load(fullfile(data_dir,['RSM1854_get_from_full_sampling_of_48_objects_',model_name,'.mat']));
    load(fullfile(data_dir,['RSM1854_get_from_full_sampling_of_48_objects_',model_name,'_splithalf.mat']));
    RSM48_triplet = RSM1854_triplet(wordposition48,wordposition48);
    RSM48_triplet_split1 = RSM1854_triplet_split1(wordposition48,wordposition48);
    RSM48_triplet_split2 = RSM1854_triplet_split2(wordposition48,wordposition48);
    RDM48_triplet = 1- RSM48_triplet;
    RDM48_triplet_split1 = 1- RSM48_triplet_split1;
    RDM48_triplet_split2 = 1- RSM48_triplet_split2;
    save_filename = [data_dir,'/RDM48_triplet.mat'];  
    save(save_filename, 'RDM48_triplet');
    save_filename = [data_dir,'/RDM48_triplet_splithalf.mat'];  
    save(save_filename, 'RDM48_triplet_split1', 'RDM48_triplet_split2');
end


%% Load relevant data for other LLMs
model_names = {'Llama3.1'};
for model = 1:length(model_names)
    model_name = model_names{model};
    data_dir = fullfile(base_dir,['data/LLMs/',model_name]);
    load(fullfile(data_dir,['RSM1854_get_from_full_sampling_of_48_objects_',model_name,'.mat']));
    load(fullfile(data_dir,['RSM1854_get_from_full_sampling_of_48_objects_',model_name,'_splithalf.mat']));
    RSM48_triplet = RSM1854_triplet(wordposition48,wordposition48);
    RSM48_triplet_split1 = RSM1854_triplet_split1(wordposition48,wordposition48);
    RSM48_triplet_split2 = RSM1854_triplet_split2(wordposition48,wordposition48);
    RDM48_triplet = 1- RSM48_triplet;
    RDM48_triplet_split1 = 1- RSM48_triplet_split1;
    RDM48_triplet_split2 = 1- RSM48_triplet_split2;
    save_filename = [data_dir,'/RDM48_triplet.mat'];  
    save(save_filename, 'RDM48_triplet');
    save_filename = [data_dir,'/RDM48_triplet_splithalf.mat'];  
    save(save_filename, 'RDM48_triplet_split1', 'RDM48_triplet_split2');
end

%% Load relevant data for other LLMs
model_names = {'Qwen2_VL'};
for model = 1:length(model_names)
    model_name = model_names{model};
    data_dir = fullfile(base_dir,['data/MLLMs/',model_name]);
    load(fullfile(data_dir,['RSM1854_get_from_full_sampling_of_48_objects_',model_name,'.mat']));
    load(fullfile(data_dir,['RSM1854_get_from_full_sampling_of_48_objects_',model_name,'_splithalf.mat']));
    RSM48_triplet = RSM1854_triplet(wordposition48,wordposition48);
    RSM48_triplet_split1 = RSM1854_triplet_split1(wordposition48,wordposition48);
    RSM48_triplet_split2 = RSM1854_triplet_split2(wordposition48,wordposition48);
    RDM48_triplet = 1- RSM48_triplet;
    RDM48_triplet_split1 = 1- RSM48_triplet_split1;
    RDM48_triplet_split2 = 1- RSM48_triplet_split2;
    save_filename = [data_dir,'/RDM48_triplet.mat'];  
    save(save_filename, 'RDM48_triplet');
    save_filename = [data_dir,'/RDM48_triplet_splithalf.mat'];  
    save(save_filename, 'RDM48_triplet_split1', 'RDM48_triplet_split2');
end



