function [acc_spose, predlabel, catlabels] = predict_category_for_layer_fea(embedding, base_dir, data_dir,variable_dir,categories27)
base_dir = fullfile(base_dir,'data');
load(fullfile(base_dir,'category_mat_manual.mat'))
load(fullfile(base_dir,'typicality_data27.mat'))
load(fullfile(variable_dir,'words.mat'))
% load sense vectors
load(fullfile(base_dir,'sensevec_augmented_with_wordvec.mat'))
embedding = load(fullfile(data_dir, embedding));
embedding = embedding.data;

catmat = category_mat_manual;
categories = categories27;
catword = words; 

% exclude objects that are part of multiple categories (unless subcategory)

% remove:
% - bird 2
% - clothing accessory 5
% - dessert 7
% - drink 8
% - fruit 11 
% - insect 14
% - vegetable 25

rmcat = [2 5 7 8 11 14 25];
categories(rmcat) = [];
catmat(:,rmcat) = [];

% set non-unique objects to 0
catmat(sum(catmat,2)>1,:) = false;

% now remove categories with too few items (less than 10)
categories([9 10]) = [];
catmat(:,[9 10]) = [];


% now reduce to relevant objects
rmind = ~any(catmat,2);
spose_embedding66_small = embedding;
spose_embedding66_small(rmind,:) = [];
catmat(rmind,:) = [];

% set label
catlabels = sum(catmat.*repmat(1:length(categories),length(catmat),1),2);


%% Use euclidean distance to centroid with leave-one-out

clear predlabel
for i_obj = 1:length(catlabels)
    
    testind = i_obj;
    trainind = setdiff(1:length(catlabels),i_obj);
    testlabel = catlabels(testind);
    trainlabel = catlabels(trainind);
    x_train = spose_embedding66_small(trainind,:);
    x_test = spose_embedding66_small(testind,:);
    
    clear mcatvec
    for i_cat = 1:length(categories)
        mcatvec(:,i_cat) = nanmean(x_train(trainlabel==i_cat,:)); % generate centroid
    end

%     tmp = 1-squareformq(pdist([mcatvec';x_test],'cos'));
%     catpred = tmp(1:end-1,end);
%     [~,predlabel_cos(i_obj,1)] = max(catpred);
    tmp = 1-squareformq(pdist([mcatvec';x_test],'euc'));
    catpred = tmp(1:end-1,end);    
    [~,predlabel(i_obj,1)] = max(catpred);
end

fprintf('Accuracy for layer feature: %2.2f\n',100*mean(predlabel==catlabels))

acc_spose=100*mean(predlabel==catlabels);

%% Repeat for semantic embedding

% get the prototype for each category
sensevec_reduc = sensevec_augmented;
sensevec_reduc(rmind,:) = [];

clear predlabel_semantic
for i_obj = 1:length(catlabels)
    
    testind = i_obj;
    trainind = setdiff(1:length(catlabels),i_obj);
    testlabel = catlabels(testind);
    trainlabel = catlabels(trainind);
    x_train = sensevec_reduc(trainind,:);
    x_test = sensevec_reduc(testind,:);
    
    clear mcatvec
    for i_cat = 1:length(categories)
        mcatvec(:,i_cat) = nanmean(x_train(trainlabel==i_cat,:)); % get centroid
    end

    tmp = 1-squareformq(pdist([mcatvec';x_test],'euc'));
    catpred = tmp(1:end-1,end);    
    [~,predlabel_semantic(i_obj,1)] = max(catpred);
end

fprintf('Accuracy for word embedding: %2.2f\n',100*mean(predlabel_semantic==catlabels))