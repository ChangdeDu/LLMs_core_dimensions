
% run this script from where it is located
base_dir = pwd;
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% Load relevant data
% load 48 object RDM
data_dir = fullfile(base_dir,'data/LLMs/Llama3.1');
% data_dir = fullfile(base_dir,'data/Humans');
load(fullfile(data_dir,'RDM48_triplet.mat'));
RDM48_triplet_behavior=RDM48_triplet;
load(fullfile(data_dir, 'RDM48_triplet_splithalf.mat'));
RDM48_triplet_split1_behavior = RDM48_triplet_split1;
RDM48_triplet_split2_behavior = RDM48_triplet_split2;


data_dir = fullfile(base_dir,'data/DNNs/LLama3-1-8B');
load(fullfile(data_dir,'RDM48_triplet.mat'));
RDM48_triplet_cosine=RDM48_triplet;
load(fullfile(data_dir, 'RDM48_triplet_splithalf.mat'));
RDM48_triplet_split1_cosine = RDM48_triplet_split1;
RDM48_triplet_split2_cosine = RDM48_triplet_split2;
%% Figure 2d: Predict behavior and similarity
dosave = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compare similarity from model internal feature to similarity in model behavior %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compare to "true" similarity
% r48 = corr(squareformq(1-RDM48_triplet_cosine),squareformq(1-RDM48_triplet_behavior), 'Type', 'Spearman')
[r48, p_value] = corr(squareformq(1-RDM48_triplet_cosine),squareformq(1-RDM48_triplet_behavior), 'Type', 'Pearson');


% run 1000 bootstrap samples for confidence intervals, bootstrap cannot 
% be done across objects because otherwise it's biased
rng(2)
rnd = randi(nchoosek(48,2),nchoosek(48,2),1000);
c1 = squareformq(1-RDM48_triplet_cosine);
c2 = squareformq(1-RDM48_triplet_behavior);
for i = 1:1000
    r48_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
end
r48_ci95_lower = tanh(atanh(r48) - 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI
r48_ci95_upper = tanh(atanh(r48) + 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI

r48_cc = corr(squareformq(RDM48_triplet_split1_cosine),squareformq(RDM48_triplet_split2_cosine));
r48_bb = corr(squareformq(RDM48_triplet_split1_behavior),squareformq(RDM48_triplet_split2_behavior));

numerator = r48;
denominator = sqrt(r48_cc * r48_bb);
rho = numerator / denominator

ci_lower = r48_ci95_lower + (rho-r48)
ci_upper = r48_ci95_upper + (rho-r48)


h = figure;
h.Position(3:4) = [1200 568];
ha = subtightplot(1,3,1);
ha.Position(1) = ha.Position(1)-0.04;
imagesc(1-RDM48_triplet_cosine,[0 1])
colormap(viridis)
hold on
text(24,-4,'Cosine RSM','HorizontalAlignment','center','FontSize',14);
axis off square tight
% xlabel('48 diverse objects')
text(25, 52, '48 diverse objects', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);

hb = subtightplot(1,3,2);
hb.Position(1) = hb.Position(1)-0.035;
imagesc(1-RDM48_triplet_behavior,[0 1])
text(24,-4,'Behavioral RSM','HorizontalAlignment','center','FontSize',14);
colormap(viridis)
axis off square tight
% hc = colorbar('Position', [0.52 0.1 0.02 0.8]); % Add a common colorbar
hc = colorbar('Location', 'south','Position', [0.376 0.145 0.245 0.03],'FontSize',13); % Add a horizontal colorbar


hc = subtightplot(1,3,3);
hc.Position(1) = hc.Position(1)+0.045;
plot(squareformq(1-RDM48_triplet_cosine),squareformq(1-RDM48_triplet_behavior),'o','MarkerFaceColor',[.2 .6 .4],'MarkerEdgeColor','none','MarkerSize',3);
hold on
plot([0 1],[0 1],'k')
axis square tight
set(gca,'FontSize',13) 
xlabel('Cosine similarity','FontSize', 16)
ylabel('Behavioral similarity','FontSize', 16)
% title(hc, 'Model fit','FontSize', 14);

legend(['R = ',num2str(round(r48, 2))],'Location','NorthWest','FontSize',13)

imageSize = [800 400]; % Adjust the size as needed
set(h, 'PaperUnits', 'inches', 'PaperSize', imageSize/100, 'PaperPosition', [0 0 imageSize/100]);

if dosave
    saveas(h, 'rsm_compare_llama3_1.pdf', 'pdf');
end



