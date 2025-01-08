
% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data

% load embedding
spose_embedding = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
% get dot product (i.e. proximity)
dot_product66 = spose_embedding*spose_embedding';
% load 48 object RDM
load(fullfile(data_dir,'RDM48_triplet.mat'));
%% Load smaller version of words for each image
load(fullfile(variable_dir,'words.mat'))
load(fullfile(variable_dir,'words48.mat'))

[~,~,wordposition48] = intersect(words48,words,'stable');
save 'data/wordposition48' wordposition48
%% Figure 2d: Predict behavior and similarity
dosave = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compare similarity from model to similarity in behavior %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for that focus on those 48 objects only rather than the entire matrix
tic
esim = exp(dot_product66);
cp = zeros(1854,1854);
ctmp = zeros(1,1854);
for i = 1:1854
    for j = i+1:1854
        ctmp = zeros(1,1854);
        for k_ind = 1:length(wordposition48)
            k = wordposition48(k_ind);
            if k == i || k == j, continue, end
            ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
        end
        cp(i,j) = sum(ctmp); % run sum first, divide all by 48 later
    end
end
toc
cp = cp/48; % complete the mean
cp = cp+cp'; % symmetric
cp(logical(eye(size(cp)))) = 1;

spose_sim48 = cp(wordposition48,wordposition48);

% compare to "true" similarity
r48 = corr(squareformq(spose_sim48),squareformq(1-RDM48_triplet))

[R, p_value] = corr(squareformq(spose_sim48), squareformq(1-RDM48_triplet), 'Type', 'Pearson');


% run 1000 bootstrap samples for confidence intervals, bootstrap cannot 
% be done across objects because otherwise it's biased
rng(2)
rnd = randi(nchoosek(48,2),nchoosek(48,2),1000);
c1 = squareformq(spose_sim48);
c2 = squareformq(1-RDM48_triplet);
for i = 1:1000
    r48_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
end
r48_ci95_lower = tanh(atanh(r48) - 1.96*std(atanh(r48_boot),[],2)) % reflects 95% CI
r48_ci95_upper = tanh(atanh(r48) + 1.96*std(atanh(r48_boot),[],2)) % reflects 95% CI


h = figure;
h.Position(3:4) = [1200 568];
ha = subtightplot(1,3,1);
ha.Position(1) = ha.Position(1)-0.04;
imagesc(spose_sim48,[0 1])
colormap(viridis)
hold on
text(24,-4,'Predicted LLM RSM','HorizontalAlignment','center','FontSize',14);
axis off square tight
% xlabel('48 diverse objects')
text(25, 52, '48 diverse objects', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);

hb = subtightplot(1,3,2);
hb.Position(1) = hb.Position(1)-0.035;
imagesc(1-RDM48_triplet,[0 1])
text(24,-4,'Measured LLM RSM','HorizontalAlignment','center','FontSize',14);
colormap(viridis)
axis off square tight
% hc = colorbar('Position', [0.52 0.1 0.02 0.8]); % Add a common colorbar
hc = colorbar('Location', 'south','Position', [0.376 0.145 0.245 0.03],'FontSize',13); % Add a horizontal colorbar


hc = subtightplot(1,3,3);
hc.Position(1) = hc.Position(1)+0.045;
plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[.2 .6 .4],'MarkerEdgeColor','none','MarkerSize',3);
% plot(squareformq(spose_sim48),squareformq(1-RDM48_triplet),'o','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','none','MarkerSize',3);
hold on
plot([0 1],[0 1],'k')
axis square tight
set(gca,'FontSize',13) 
xlabel('predicted similarity','FontSize', 16)
ylabel('measured similarity','FontSize', 16)
title(hc, 'Model fit','FontSize', 14);

legend(['R = ',num2str(round(r48, 2))],'Location','NorthWest','FontSize',13)

imageSize = [800 400]; % Adjust the size as needed
set(h, 'PaperUnits', 'inches', 'PaperSize', imageSize/100, 'PaperPosition', [0 0 imageSize/100]);

if dosave
    saveas(h, 'rsm_compare_chatgpt.pdf', 'pdf');
end



