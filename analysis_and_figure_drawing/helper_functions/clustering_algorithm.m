function [ind,fams_sorted] = clustering_algorithm(n_iter,cutoff,sparsedims49)

% This is our own clustering algorithm that creates "families" of objects
% based on the most dominant dimensions. This is only for visualization
% purposes.

[s,i] = sort(sparsedims49,2);

% Now pick the top n_iter dimensions for determining a family
j = sort(i(:,end-n_iter+1:end),2); % sorting is backwards, so we start with the lowest number, which should also give us the highest weight

fams = j(:,n_iter);
for i_iter = 1:n_iter-1
    k_iter = n_iter-i_iter;
    fams = fams + j(:,i_iter)*10^(2*(k_iter));
end
% fams = j(:,1)*100000000 + j(:,2)*1000000 + j(:,3)*10000 + j(:,4)*100 + j(:,5);

for i_iter = 1:n_iter
    
    ufams = unique(fams);
    h = hist(fams',ufams);

% now we take all the families with less than 5 entries and ignore the 3rd
% dimension and put them together into a class with only the first
% dimension (i.e. 99 at the end)
for ifam = 1:length(ufams)
    if h(ifam)<cutoff
        mult = 10^(2*i_iter); % TODO: not sure this is correct, always needs to be shifted one forward (for first, we want 100, for second 10000, etc.)
        fams(fams==ufams(ifam)) = floor(fams(fams==ufams(ifam))/mult)*mult + 99*10^(2*(i_iter-1)); % add 99*mult so it ends up at the end
    end
end

end




ufams = unique(fams);

h = hist(fams',ufams);

% In the last step, we'll take all clusters < 5 and put them together,
% adding 99 to the front
for ifam = 1:length(ufams)
    if h(ifam)<cutoff
         fams(fams==ufams(ifam)) =  fams(fams==ufams(ifam))+ 9e6;
    end
end

[fams_sorted,ind] = sort(fams);

% figure, imagesc(sparsedims49(ind,:)*sparsedims49(ind,:)')




%% original code
% 
% j = i(:,end-n_iter+1:end); % try without sorting
% % j = sort(i(:,end-n_iter+1:end),2); % sorting is backwards, so we start with the lowest number, which should also give us the highest weight
% 
% fams = j(:,1)*10000 + j(:,2)*100 + j(:,3);
% 
% ufams = unique(fams);
% 
% h = hist(fams',ufams);
% 
% % now we take all the families with less than 5 entries and ignore the 3rd
% % dimension and put them together into a class with only the first
% % dimension (i.e. 99 at the end)
% 
% for ifam = 1:length(ufams)
%     if h(ifam)<cutoff
%         fams(fams==ufams(ifam)) = floor(fams(fams==ufams(ifam))/100)*100 + 99; % add 99 so it ends up at the end
%     end
% end
% 
% 
% 
% % now we repeat this with the first and second dimension
% 
% ufams = unique(fams);
% 
% h = hist(fams',ufams);
% 
% % now we take all the families with less than 5 entries and ignore the 3rd
% % dimension and put them together into a class with only the first
% % dimension (i.e. 99 at the end)
% 
% for ifam = 1:length(ufams)
%     if h(ifam)<cutoff
%         fams(fams==ufams(ifam)) = floor(fams(fams==ufams(ifam))/10000)*10000 + 9900; % add 9900 so it ends up at the end
%     end
% end
% 
% 
% ufams = unique(fams);
% 
% h = hist(fams',ufams);
% 
% % In the last step, we'll take all clusters < 5 and put them together,
% % adding 99 to the front
% for ifam = 1:length(ufams)
%     if h(ifam)<cutoff
%          fams(fams==ufams(ifam)) =  fams(fams==ufams(ifam))+ 9e6;
%     end
% end
% 
% [~,ind] = sort(fams);
% 
% figure, imagesc(sparsedims49(ind,:)*sparsedims49(ind,:)')
% 

%% run clustering on the three most dominant dimensions

