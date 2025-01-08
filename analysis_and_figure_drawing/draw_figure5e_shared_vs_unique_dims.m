clc;clear;

% run this script from where it is located
base_dir = pwd;

variable_dir = fullfile(base_dir,'data/variables');

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data

% load embedding
data_dir = fullfile(base_dir,'data/LLMs/ChatGPT-3.5');
spose_embedding_chatgpt = load(fullfile(data_dir,'spose_embedding_66d_sorted_chatgpt.txt'));
% load embedding
data_dir = fullfile(base_dir,'data/MLLMs/Gemini_Pro_Vision');
spose_embedding_gemini = load(fullfile(data_dir,'spose_embedding_66d_sorted_gemini.txt'));
% load embedding
data_dir = fullfile(base_dir,'data/Humans/');
spose_embedding_human = load(fullfile(data_dir,'spose_embedding_66d_sorted_humans.txt'));


% 假设 A, B, C 已经加载到工作空间中，每个都是 1854x66 的矩阵
A = spose_embedding_human;
B = spose_embedding_chatgpt;
C = spose_embedding_gemini;

%% get adapted c for AB
c = corr(spose_embedding_human,spose_embedding_chatgpt);
% now let's sort dimensions by maximal cross-correlation
[~,ii] = max(c);
% remove ones that have already appeared
for i = 2:66
    if any(ii(1:i-1)==ii(i))
        ii(i) = 100;
    end
end
[~,si] = sort(ii);

% now remove original cross-correlation matrix to find differences
c_base = [(corr(spose_embedding_human) - eye(66))];
c_adapted_AB = c(:,si)-c_base;

%% get adapted c for AC
c = corr(spose_embedding_human,spose_embedding_gemini);
% now let's sort dimensions by maximal cross-correlation
[~,ii] = max(c);
% remove ones that have already appeared
for i = 2:66
    if any(ii(1:i-1)==ii(i))
        ii(i) = 100;
    end
end
[~,si] = sort(ii);

% now remove original cross-correlation matrix to find differences
c_base = [(corr(spose_embedding_human) - eye(66))];
c_adapted_AC = c(:,si)-c_base;


%% get adapted c for BC
c = corr(spose_embedding_chatgpt,spose_embedding_gemini);
% now let's sort dimensions by maximal cross-correlation
[~,ii] = max(c);
% remove ones that have already appeared
for i = 2:66
    if any(ii(1:i-1)==ii(i))
        ii(i) = 100;
    end
end
[~,si] = sort(ii);

% now remove original cross-correlation matrix to find differences
c_base = [(corr(spose_embedding_chatgpt) - eye(66))];
c_adapted_BC = c(:,si)-c_base;



%% 设置相关性阈值
threshold = 0.2;

% 初始化结果列表
result_A = zeros(size(A,2), 3, 'logical');
result_B = zeros(size(B,2), 3, 'logical');
result_C = zeros(size(C,2), 3, 'logical');

% 初始化待处理的维度
dims_A = 1:size(A,2);
dims_B = 1:size(B,2);
dims_C = 1:size(C,2);
for i = dims_A
    result_A(i, 1) = 1;  % A 总是包含自己的维度
end
for i = dims_B
    result_B(i, 2) = 1;  % B 总是包含自己的维度
end
for i = dims_C
    result_C(i, 3) = 1;  % C 总是包含自己的维度
end

% 遍历 A 的列
for i = dims_A
    % 检查与 B 的相关性
    for j = dims_B
        if c_adapted_AB(i, j) >= threshold
            result_A(i, 2) = 1;
            result_B(j, 2) = 0;
            dims_B(dims_B == j) = [];  % 从 B 中移除此维度
            break;
        end
    end
    
    % 检查与 C 的相关性
    for j = dims_C
        if c_adapted_AC(i, j) >= threshold
            result_A(i, 3) = 1;
            result_C(j, 3) = 0;
            dims_C(dims_C == j) = [];  % 从 C 中移除此维度
            break;
        end
    end
end


% 遍历剩余的 B 的列
for i = dims_B
    % 检查与剩余 C 的相关性
    for j = dims_C
        if c_adapted_BC(i, j) >= threshold
            result_B(i, 3) = 1;
            result_C(j, 3) = 0;
            dims_C(dims_C == j) = [];  % 从 C 中移除此维度
            break;
        end
    end
end

% 合并结果
result = [result_A; result_B; result_C];

% 移除全零行
result = result(any(result, 2), :);

% 统计结果
shared_ABC = sum(all(result, 2));
shared_AB = sum(result(:, 1) & result(:, 2) & ~result(:, 3));
shared_AC = sum(result(:, 1) & ~result(:, 2) & result(:, 3));
shared_BC = sum(~result(:, 1) & result(:, 2) & result(:, 3));
unique_A = sum(result(:, 1) & ~result(:, 2) & ~result(:, 3));
unique_B = sum(~result(:, 1) & result(:, 2) & ~result(:, 3));
unique_C = sum(~result(:, 1) & ~result(:, 2) & result(:, 3));

% 输出结果
fprintf('共有维度 (A, B, C): %d\n', shared_ABC);
fprintf('共有维度 (A, B): %d\n', shared_AB);
fprintf('共有维度 (A, C): %d\n', shared_AC);
fprintf('共有维度 (B, C): %d\n', shared_BC);
fprintf('A 独有维度: %d\n', unique_A);
fprintf('B 独有维度: %d\n', unique_B);
fprintf('C 独有维度: %d\n', unique_C);
fprintf('结果矩阵行数: %d\n', size(result, 1));


setName={'Human','LLM','MLLM'};
Data=result;


bar1Color=[0,0,245;245,0,0]./255;
bar2Color=[.2 .4 .6;
          .2 .6 .4;
          .6 .2 .4];
lineColor=[61,58,61]./255;


vennColor=[.2 .6 .4;
           .6 .2 .4;
           .2 .4 .6;];
mylabels =  [unique_A, unique_B, unique_C, shared_AB, shared_AC, shared_BC, shared_ABC]; % A, B, C, A&B, A&C, B&C and A&B&C. 
fig = venn(3,'sets',setName,'labels',mylabels,'alpha',0.9,'edgeC',[0 0 0],'colors',vennColor,'edgeW',6);
exportgraphics(fig, 'shared_vs_specific_dims_vennplot.pdf', 'ContentType', 'vector');

%% =========================================================================
% 进行组合统计(一顿花里胡哨的操作)
pBool=abs(dec2bin((1:(2^size(Data,2)-1))'))-48;
[pPos,~]=find(((pBool*(1-Data'))|((1-pBool)*Data'))==0);
sPPos=sort(pPos);dPPos=find([diff(sPPos);1]);
pType=sPPos(dPPos);pCount=diff([0;dPPos]);
[pCount,pInd]=sort(pCount,'descend');
pType=pType(pInd);
sCount=sum(Data,1);
[sCount,sInd]=sort(sCount,'descend');
sType=1:size(Data,2);
sType=sType(sInd);
%% ========================================================================
% 构造figure及axes
fig=figure('Units','normalized','Position',[.3,.2,.63,.63],'Color',[1,1,1]);
axI=axes('Parent',fig);hold on;
set(axI,'Position',[.33,.35,.655,.61],'LineWidth',1.2,'Box','off','TickDir','out',...
    'FontName','Arial','FontSize',18,'XTick',[],'XLim',[0,length(pType)+1])
axI.YLabel.String='Intersection Size';
axI.YLabel.FontSize=20;
%
axS=axes('Parent',fig);hold on;
set(axS,'Position',[.01,.08,.245,.26],'LineWidth',1.2,'Box','off','TickDir','out',...
    'FontName','Arial','FontSize',18,'YColor','none','YLim',[.5,size(Data,2)+.5],...
    'YAxisLocation','right','XDir','reverse','YTick',[])
axS.XLabel.String='Set Size';
axS.XLabel.FontSize=20;
%
axL=axes('Parent',fig);hold on;
set(axL,'Position',[.33,.08,.655,.26],'YColor','none','YLim',[.5,size(Data,2)+.5],'XColor','none','XLim',axI.XLim)
%% ========================================================================
% 相交关系统计图 -----------------------------------------------------------
barHdlI=bar(axI,pCount);
barHdlI.EdgeColor='none';
if size(bar1Color,1)==1
    bar1Color=[bar1Color;bar1Color];
end
tx=linspace(0,1,size(bar1Color,1))';
ty1=bar1Color(:,1);ty2=bar1Color(:,2);ty3=bar1Color(:,3);
tX=linspace(0,1,length(pType))';
bar1Color=[interp1(tx,ty1,tX,'pchip'),interp1(tx,ty2,tX,'pchip'),interp1(tx,ty3,tX,'pchip')];
barHdlI.FaceColor='flat';
for i=1:length(pType)
    barHdlI.CData(i,:)=bar1Color(i,:);
end
text(axI,1:length(pType),pCount,string(pCount),'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontName','Arial','FontSize',18,'Color',[61,58,61]./255)
% 集合统计图 ---------------------------------------------------------------
barHdlS=barh(axS,sCount,'BarWidth',.6);
barHdlS.EdgeColor='none';
barHdlS.BaseLine.Color='none';
for i=1:size(Data,2)
    annotation('textbox',[(axS.Position(1)+axS.Position(3)+axI.Position(1))/2-.02,...
        axS.Position(2)+axS.Position(4)./size(Data,2).*(i-.5)-.02,.04,.04],...
        'String',setName{sInd(i)},'HorizontalAlignment','center','VerticalAlignment','middle',...
        'FitBoxToText','on','LineStyle','none','FontName','Arial','FontSize',18)
end
if size(bar2Color,1)==1
    bar2Color=[bar2Color;bar2Color];
end
tx=linspace(0,1,size(bar2Color,1))';
ty1=bar2Color(:,1);ty2=bar2Color(:,2);ty3=bar2Color(:,3);
tX=linspace(0,1,size(Data,2))';
bar2Color=[interp1(tx,ty1,tX,'pchip'),interp1(tx,ty2,tX,'pchip'),interp1(tx,ty3,tX,'pchip')];
barHdlS.FaceColor='flat';
sstr{size(Data,2)}='';
for i=1:size(Data,2)
    barHdlS.CData(i,:)=bar2Color(i,:);
    sstr{i}=[num2str(sCount(i)),' '];
end
text(axS,sCount,1:size(Data,2),sstr,'HorizontalAlignment','right',...
    'VerticalAlignment','middle','FontName','Arial','FontSize',18,'Color',[61,58,61]./255)
% 绘制关系连线 ---------------------------------------------------------------
patchColor=[248,246,249;255,254,255]./255;
for i=1:size(Data,2)
    fill(axL,axI.XLim([1,2,2,1]),[-.5,-.5,.5,.5]+i,patchColor(mod(i+1,2)+1,:),'EdgeColor','none')
end
[tX,tY]=meshgrid(1:length(pType),1:size(Data,2));
plot(axL,tX(:),tY(:),'o','Color',[233,233,233]./255,...
    'MarkerFaceColor',[233,233,233]./255,'MarkerSize',14);
for i=1:length(pType)
    tY=find(pBool(pType(i),:));
    oY=zeros(size(tY));
    for j=1:length(tY)
        oY(j)=find(sType==tY(j));
    end
    tX=i.*ones(size(tY));
    plot(axL,tX(:),oY(:),'-o','Color',lineColor(1,:),'MarkerEdgeColor','none',...
        'MarkerFaceColor',lineColor(1,:),'MarkerSize',14,'LineWidth',2);
end

exportgraphics(fig, 'shared_vs_specific_dims_upsetplot.pdf', 'ContentType', 'vector');
% close(fig);