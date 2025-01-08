% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'data');
variable_dir = fullfile(base_dir,'data/variables');

roi_names = {'EarlyVis','RSC', 'OPA', 'EBA', 'FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca'}; 
subj_id = [1,2,5,7];
embedding_names = {'LLM','VLM','Human','CLIPvision','CLIPtext'};
embedding_names_label = {'LLM','MLLM','Human','CLIPvision','CLIPtext'};


LLM_scores = zeros(length(roi_names), length(subj_id));
MLLM_scores = zeros(length(roi_names), length(subj_id));
Human_scores = zeros(length(roi_names), length(subj_id));
CLIPvision_scores = zeros(length(roi_names), length(subj_id));
CLIPtext_scores = zeros(length(roi_names), length(subj_id));
for i = 1:length(subj_id)
    subj = ['Subject_', num2str(subj_id(i))];
    load(fullfile(data_dir,['ROIs/roi_spose_rsa_scores_', subj, '.mat']));
    LLM_scores(:,i)         = data(1,:);
    MLLM_scores(:,i)        = data(2,:);
    Human_scores(:,i)       = data(3,:);
    CLIPvision_scores(:,i)  = data(4,:);
    CLIPtext_scores(:,i)    = data(5,:);
end


fig = figure('Units','normalized','Position',[.2,.3,.56,.45],'Color','w');

rng(24)
DataA = LLM_scores;
DataB = MLLM_scores;
DataC = Human_scores;
DataD = CLIPvision_scores;
DataE = CLIPtext_scores;
meanData = [mean(DataA,2), mean(DataB,2), mean(DataC,2), mean(DataD,2), mean(DataE,2)];

CList = [.2 .6 .4; 
         .6 .2 .4; 
         .2 .4 .6;
         .4 .2 .8; 
         .8 .5 .3];

% 横坐标标签文本
NameList = roi_names;

hold on
% 绘制柱状图 ---------------------------------------------------------------
barHdl = bar(meanData,'EdgeColor','none','FaceAlpha',1,'BarWidth',.7);
% 修改配色
barHdl(1).FaceColor = CList(1,:);
barHdl(2).FaceColor = CList(2,:);
barHdl(3).FaceColor = CList(3,:);
barHdl(4).FaceColor = CList(4,:);
barHdl(5).FaceColor = CList(5,:);
% 绘制散点图 ---------------------------------------------------------------
XA = barHdl(1).XEndPoints.'*ones(1,size(DataA,2));
scatter(XA(:),DataA(:),15,'filled','o','MarkerEdgeColor','k','LineWidth',.8,...
                      'CData',CList(1,:),'XJitter','rand','XJitterWidth',0.05)
XB = barHdl(2).XEndPoints.'*ones(1,size(DataB,2));
scatter(XB(:),DataB(:),15,'filled','o','MarkerEdgeColor','k','LineWidth',.8,...
                      'CData',CList(2,:),'XJitter','rand','XJitterWidth',0.05)
XC = barHdl(3).XEndPoints.'*ones(1,size(DataC,2));
scatter(XC(:),DataC(:),15,'filled','o','MarkerEdgeColor','k','LineWidth',.8,...
                      'CData',CList(3,:),'XJitter','rand','XJitterWidth',0.05)
XD = barHdl(4).XEndPoints.'*ones(1,size(DataD,2));
scatter(XD(:),DataD(:),15,'filled','o','MarkerEdgeColor','k','LineWidth',.8,...
                      'CData',CList(4,:),'XJitter','rand','XJitterWidth',0.05)
XE = barHdl(5).XEndPoints.'*ones(1,size(DataE,2));
scatter(XE(:),DataE(:),15,'filled','o','MarkerEdgeColor','k','LineWidth',.8,...
                      'CData',CList(5,:),'XJitter','rand','XJitterWidth',0.05)
% 绘制误差棒 ---------------------------------------------------------------
errorbar(barHdl(1).XEndPoints,meanData(:,1),std(DataA,0,2),'vertical',...
    'LineStyle','none','LineWidth',1,'Color','k')
errorbar(barHdl(2).XEndPoints,meanData(:,2),std(DataB,0,2),'vertical',...
    'LineStyle','none','LineWidth',1,'Color','k')
errorbar(barHdl(3).XEndPoints,meanData(:,3),std(DataC,0,2),'vertical',...
    'LineStyle','none','LineWidth',1,'Color','k')
errorbar(barHdl(4).XEndPoints,meanData(:,4),std(DataD,0,2),'vertical',...
    'LineStyle','none','LineWidth',1,'Color','k')
errorbar(barHdl(5).XEndPoints,meanData(:,5),std(DataE,0,2),'vertical',...
    'LineStyle','none','LineWidth',1,'Color','k')
% 坐标区域简单修饰
ax = gca;
ax.XTick = 1:length(barHdl(1).XEndPoints);
ax.XTickLabel = NameList(:);
ax.FontName = 'Arial';
ax.FontWeight = 'bold';
ax.FontSize = 11;
ax.XTickLabelRotation = 35;
ax.LineWidth = 1.5;

ylabel('SPoSE RSA score', 'FontSize', 15);
xlabel('Brain ROIs', 'FontSize', 14);
legend(embedding_names_label, 'Location', 'best', 'FontSize', 13);
title('4 subjects averaged',  'FontSize', 16);

exportgraphics(fig, 'figures/spose_rsa_barplot_scatters.pdf', 'ContentType', 'vector');
close(fig);
