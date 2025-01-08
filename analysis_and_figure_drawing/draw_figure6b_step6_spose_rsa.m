% 定义基础目录
base_dir = pwd;
variable_dir = fullfile(base_dir, 'data/variables');

% 添加相关的工具箱
addpath(base_dir)
addpath(genpath(fullfile(base_dir, 'helper_functions')))

% 定义共享的索引
shared_1k = 982;
% 加载和处理不同的嵌入

spose_sim_shared_1k_chatgpt = calculateSimilarityMatrix(fullfile(base_dir, 'data/LLMs/ChatGPT-3.5'), 'spose_embedding_nsd_shared_1k_predicted_from_chatgpt.txt', shared_1k);
spose_sim_shared_1k_gemini = calculateSimilarityMatrix(fullfile(base_dir, 'data/MLLMs/Gemini_Pro_Vision'), 'spose_embedding_nsd_shared_1k_predicted_from_gemini.txt', shared_1k);
spose_sim_shared_1k_human = calculateSimilarityMatrix(fullfile(base_dir, 'data/Humans/'), 'spose_embedding_nsd_shared_1k_predicted_from_humans.txt', shared_1k);
sim_shared_1k_CLIP_visual = calculateSimilarityMatrix(fullfile(base_dir, 'data/DNNs/CLIPvison_ViT_L14'), 'spose_embedding_nsd_shared_1k_predicted_from_CLIPvison_ViT_L14.txt', shared_1k);
sim_shared_1k_CLIP_language = calculateSimilarityMatrix(fullfile(base_dir, 'data/DNNs/CLIPtext_ViT_L14'), 'spose_embedding_nsd_shared_1k_predicted_from_CLIPtext_ViT_L14.txt', shared_1k);

% 初始化变量
subj_id = [1, 2, 5, 7];
roi_sets = {'EarlyVis','RSC', 'OPA', 'EBA', 'FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca'}; 
embedding_names = {'LLM', 'MLLM', 'Human', 'CLIPvision', 'CLIPtext'};
embedding_RSMs={spose_sim_shared_1k_chatgpt,spose_sim_shared_1k_gemini,spose_sim_shared_1k_human,sim_shared_1k_CLIP_visual,sim_shared_1k_CLIP_language};
tempFolder = 'data/ROIs';
if ~exist(tempFolder, 'dir')
    mkdir(tempFolder);
end

% 循环处理每个被试和ROI
for sub_id = 1:length(subj_id)
    subj = ['Subject_', num2str(subj_id(sub_id))]
    subj_scores = zeros(length(embedding_names), length(roi_sets));
    for roi_idx = 1:length(roi_sets)
        roi_name = roi_sets{roi_idx};
        data_dir = fullfile(base_dir, tempFolder, subj, roi_name);
        roi_embedding_RSM = calculateSimilarityMatrix(data_dir, 'spose_embedding_sorted_merge.txt', shared_1k);
        
        for k = 1:length(embedding_names)
            subj_scores(k,roi_idx) = corr(squareformq(embedding_RSMs{k}), squareformq(roi_embedding_RSM));
        end
    end

    data = subj_scores;
    save_filename = [tempFolder,'/roi_spose_rsa_scores_', subj,'.mat']; 
    save(save_filename, 'data'); 
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
