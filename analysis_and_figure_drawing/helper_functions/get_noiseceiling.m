function [noise_ceiling,noise_ceiling_ci95] = get_noiseceiling(NCdat)

% sort each triplet and change choice id
for i = 1:length(NCdat)
    [sorted,sortind] = sort(NCdat(i,1:3));
    NCdat(i,1:4) = [sorted find(sortind==NCdat(i,4))];
end

% get unique ID for each triplet by merging numbers
NCstr = num2cell(num2str(NCdat(:,1:3)),2);
uid = unique(NCstr);
nid = length(uid);

% get number of triplets for each
for i = 1:nid
   nNC(i) = sum(strcmp(NCstr,uid{i}));  
end

% Now run for all just to see what happens (get how many people respond the same)
for i = 1:nid
    ind = strcmp(NCstr,uid{i});
    answers = NCdat(ind,4);
    h = hist(answers,1:3);
    consistency(i,1) = max(h)/sum(h); % the best one divided by all
end

noise_ceiling = mean(consistency)*100;
noise_ceiling_ci95 = 1.96 * std(consistency)*100 / sqrt(nid);