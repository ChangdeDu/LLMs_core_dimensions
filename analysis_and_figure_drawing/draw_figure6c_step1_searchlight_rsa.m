
% run this script from where it is located
base_dir = pwd;
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes

% t-SNE from: https://lvdmaaten.github.io/tsne/#implementations
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

shared_1k=982;
re_rankindex = 1:1:shared_1k;

% 加载和处理不同的嵌入
spose_sim_shared_1k_chatgpt = calculateSimilarityMatrix(fullfile(base_dir, 'data/LLMs/ChatGPT-3.5'), 'spose_embedding_nsd_shared_1k_predicted_from_chatgpt.txt', shared_1k);
spose_sim_shared_1k_gemini = calculateSimilarityMatrix(fullfile(base_dir, 'data/MLLMs/Gemini_Pro_Vision'), 'spose_embedding_nsd_shared_1k_predicted_from_gemini.txt', shared_1k);
spose_sim_shared_1k_human = calculateSimilarityMatrix(fullfile(base_dir, 'data/Humans/'), 'spose_embedding_nsd_shared_1k_predicted_from_humans.txt', shared_1k);
sim_shared_1k_CLIP_visual = calculateSimilarityMatrix(fullfile(base_dir, 'data/DNNs/CLIPvison_ViT_L14'), 'spose_embedding_nsd_shared_1k_predicted_from_CLIPvison_ViT_L14.txt', shared_1k);
sim_shared_1k_CLIP_language = calculateSimilarityMatrix(fullfile(base_dir, 'data/DNNs/CLIPtext_ViT_L14'), 'spose_embedding_nsd_shared_1k_predicted_from_CLIPtext_ViT_L14.txt', shared_1k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compare similarity from model to similarity in brain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subj_id = [1,2,5,7];
embedding_names = {'LLM','MLLM','Human','CLIPvision','CLIPtext'};
spose_sim_shared_1k={spose_sim_shared_1k_chatgpt,spose_sim_shared_1k_gemini,spose_sim_shared_1k_human,sim_shared_1k_CLIP_visual,sim_shared_1k_CLIP_language};

tempFolder = 'data/ROIs';
if ~exist(tempFolder, 'dir')
    mkdir(tempFolder);
end

for sub_id = 1:length(subj_id)
    subj = ['Subject_', num2str(subj_id(sub_id))]
    samples_volume = h5read(['data/ROIs/', subj, '/fmri_shared_1k_all_rois_4d.h5'], '/data');
    samples_volume = permute(samples_volume, [4, 3, 2, 1]);

    samples_volume_trial_1 = h5read(['data/ROIs/', subj, '/fmri_shared_1k_all_rois_4d_trial_1.h5'], '/data');
    samples_volume_trial_1 = permute(samples_volume_trial_1, [4, 3, 2, 1]);

    samples_volume_trial_2 = h5read(['data/ROIs/', subj, '/fmri_shared_1k_all_rois_4d_trial_2.h5'], '/data');
    samples_volume_trial_2 = permute(samples_volume_trial_2, [4, 3, 2, 1]);

    samples_volume_trial_3 = h5read(['data/ROIs/', subj, '/fmri_shared_1k_all_rois_4d_trial_3.h5'], '/data');
    samples_volume_trial_3 = permute(samples_volume_trial_3, [4, 3, 2, 1]);

    sample_data = squeeze(samples_volume(1,:,:,:));
    non_zero_count = nnz(sample_data);
    
    nz_voxels = find(sample_data ~= 0);
    xyz = size(sample_data);
    subj_scores = zeros(length(embedding_names),xyz(1),xyz(2),xyz(3));
    subj_scores_unnormlized = zeros(length(embedding_names),xyz(1),xyz(2),xyz(3));
    noise_ceilings = zeros(xyz(1),xyz(2),xyz(3));
    for voxel_idx = 1:numel(nz_voxels)
        percentage = voxel_idx/length(nz_voxels)
        [x, y, z] = ind2sub(size(sample_data), nz_voxels(voxel_idx));
        sphere_voxels = [];
        sphere_voxels_trial_1  = [];
        sphere_voxels_trial_2  = [];
        sphere_voxels_trial_3  = [];
        for i = x-3:x+3
            for j = y-3:y+3
                for k = z-3:z+3
                    if i >= 1 && i <= size(sample_data, 1) && j >= 1 && j <= size(sample_data, 2) && k >= 1 && k <= size(sample_data, 3)
                        sphere_voxels = [sphere_voxels samples_volume(:, i, j, k)];
                        sphere_voxels_trial_1  = [sphere_voxels_trial_1  samples_volume_trial_1(:, i, j, k)];
                        sphere_voxels_trial_2  = [sphere_voxels_trial_2  samples_volume_trial_2(:, i, j, k)];
                        sphere_voxels_trial_3  = [sphere_voxels_trial_3  samples_volume_trial_3(:, i, j, k)];
                    end
                end
            end
        end
        
        data = sphere_voxels(re_rankindex, :);
        RSM_shared_1k_brain = corrcoef(data');

        data = sphere_voxels_trial_1(re_rankindex, :);
        RSM_shared_1k_brain_trial_1 = corrcoef(data');

        data = sphere_voxels_trial_2(re_rankindex, :);
        RSM_shared_1k_brain_trial_2 = corrcoef(data');

        data = sphere_voxels_trial_3(re_rankindex, :);
        RSM_shared_1k_brain_trial_3 = corrcoef(data');

        RSM_shared_1k_brain_mean = (RSM_shared_1k_brain_trial_1 + RSM_shared_1k_brain_trial_2 + RSM_shared_1k_brain_trial_3)/3;
        r1 = corr(squareformq(RSM_shared_1k_brain_mean), squareformq(RSM_shared_1k_brain_trial_1));
        r2 = corr(squareformq(RSM_shared_1k_brain_mean), squareformq(RSM_shared_1k_brain_trial_2));
        r3 = corr(squareformq(RSM_shared_1k_brain_mean), squareformq(RSM_shared_1k_brain_trial_3));
        noise_ceiling=max([r1,r2,r3]);
        noise_ceilings(x,y,z) = noise_ceiling;

        for k = 1:length(embedding_names)
            r = corr(squareformq(spose_sim_shared_1k{k}), squareformq(RSM_shared_1k_brain));
            r_normlized =r/noise_ceiling;
            subj_scores(k,x,y,z) = r_normlized;
            subj_scores_unnormlized(k,x,y,z) = r;
        end
    end

    save_filename = [tempFolder,'/noise_ceilings_', subj,'.mat'];  
    save(save_filename, 'noise_ceilings');  

    for model = 1:length(embedding_names)
        model_name = embedding_names(model);
        model_data = squeeze(subj_scores(model,:,:,:));
        save_filename = [tempFolder,'/rsa_scores_', subj,'_',model_name{1},'.mat']; 
        save(save_filename, 'model_data'); 
        
        model_data = squeeze(subj_scores_unnormlized(model,:,:,:));
        save_filename = [tempFolder,'/rsa_scores_unnormlized_', subj,'_',model_name{1},'.mat']; 
        save(save_filename, 'model_data'); 
    end

end

% 定义加载和处理嵌入的函数
function sim_matrix = calculateSimilarityMatrix(data_dir, filename, shared_1k)
    re_rankindex = 1:1:shared_1k;
    spose_embedding = load(fullfile(data_dir, filename));
    embedding = spose_embedding(re_rankindex, :);
    dot_product_shared_1k = embedding * embedding';
    objectposition_shared_1k = 1:shared_1k;
    esim = exp(dot_product_shared_1k);
    cp = zeros(shared_1k, shared_1k);
    ctmp = zeros(1, shared_1k);
    for i = 1:shared_1k
        for j = i+1:shared_1k
            ctmp = zeros(1, shared_1k);
            for k_ind = 1:length(objectposition_shared_1k)
                k = objectposition_shared_1k(k_ind);
                if k == i || k == j, continue; end
                ctmp(k) = esim(i, j) / (esim(i, j) + esim(i, k) + esim(j, k));
            end
            cp(i, j) = sum(ctmp); % run sum first, divide all by shared_1k later
        end
    end
    cp = cp / shared_1k; % complete the mean
    cp = cp + cp'; % symmetric
    cp(logical(eye(size(cp)))) = 1;
    sim_matrix = cp;
end
