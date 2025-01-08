% run this script from where it is located
base_dir = pwd;
data_dir = fullfile(base_dir,'GPT3.5vsGPT4');

addpath(fullfile(data_dir,'util'))
load(fullfile(data_dir,'similarities.mat'));

%% settings
compPairs = {'ChatGPT-3.5','GPT-4'};

fsize = 17;
msize = 7;%1.5;
errorbarLineWidth = 3;
close all
h = ffigure;

noise_ceiling = 67.2208;
noise_ceiling_ci95 = 1.0392;

%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
% first plot noise ceiling
wid = 8;
x = 1+ [-wid wid wid -wid];
nc1 = noise_ceiling+noise_ceiling_ci95;
nc2 = noise_ceiling-noise_ceiling_ci95;
y = [nc1 nc1 nc2 nc2];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
text(1.5, 65, 'Noise celling', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', fsize);

hold on
b=bar(mean(similarities,1), 'edgecolor','none', 'FaceColor', [.2 .6 .5]);
errorbar(mean(similarities,1)',ciestim3(similarities,1,0.95,'onesided')','.k','LineWidth', errorbarLineWidth);
dif = 0.2;
xloc = repmat((1:length(compPairs))+dif,size(similarities,1),1);
xloc = xloc+repmat(randn(size(xloc,1),1)/50,1,2);
plot(xloc',similarities','-o','Color',[1,1,1]*0.4,'MarkerEdgeColor',[1,1,1]*0.2,'MarkerFaceColor',[.2 .6 .5],'MarkerSize',msize);
set(gca,'FontSize',fsize)
ylabel(sprintf('Behavioral consistency with human (%%)'), 'Position', [-0.7, 33.3, 1]);
axname({'',''},1)

for i = 1:length(compPairs)
    xPos = i;
    yPos = 20;
    text(xPos, yPos, compPairs{i}, 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', fsize+2, 'Color', 'white');
end
ylim([10,70])
xlim([0,2.8])
xticklabel_rotate([],45)
uistack(gca, 'bottom');
% hb = plot(x(1:2),[33.3333 33.3333],'--r','LineWidth',3);
hh = hline(10:10:70,'-k');
set(hh,'Color',[1,1,1]*0.8)

grid off;

pbaspect([1.0 3.0 1]);
ffine(h)
fname = 'figure_S1.pdf';
fprintf('%s\n',savprint(h,fname));

%%
close all
