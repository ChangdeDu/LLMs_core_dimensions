function [rmind,n_rm,n_unique] = check_for_cheaters(str)

% here we are not removing participants completely who have more than 5 too fast HITs

for i = 1:length(str)
    if length(str(i).RT)<20
        str(i).RT = nan(1,20);
        str(i).choice = nan(1,20);
    end
    if length(str(i).RT)>20
        str(i).RT = str(i).RT(1:20);
        str(i).choice = str(i).choice(1:20);
    end
end

RTs = (vertcat(str.RT));
RTs(isnan(RTs)) = 99999;
RTsorted = sort(RTs,2);

rmind = intersect(find(RTsorted(:,5)<800),find(RTsorted(:,10)<1100));

% only label participants that cheat at least 5x
WorkerIds = {str(rmind).WorkerId}';
[uId,~,ind] = unique(WorkerIds);
clear indind
for i = 1:length(uId)
    indind(i) = sum(ind==i)>=5;
end
% find hits of rmind associated with those workers
uId = uId(indind);



% Within each worker who did more than 10 HITs we can check reliably if they cheated based on their randomness
allWorkerIds = {str.WorkerId}';
uniqueIds = unique(allWorkerIds);

allinds = {};
check = zeros(0,3);
allmean = zeros(1e5,3); ct = 0;
alldevp = zeros(1e5,1);
alldevq = zeros(1e5,1);
% warning('off','stats:runstest:ValuesOmitted')
for i = 1:length(uniqueIds)
    currind = find(strcmp(allWorkerIds,uniqueIds{i}));
    allinds{i} = currind;
    if length(currind)>=5
        choices = [str(currind).choice];
        choices(isnan(choices)) = [];
%         cutoff = binoinv([0.005 0.995],length(choices),1/3)/length(choices);
        if any([mean(choices==1) mean(choices==2) mean(choices==3)]>0.40) || any([mean(choices==1) mean(choices==2) mean(choices==3)]<0.27)
            check(i,1) = 0;
        else
            check(i,1) = 1;
        end
        % probability of switching from one to another should be same for any
%         h = runstest(choices',mean(choices),'alpha',0.01);
        [h,devp,devq] = check_transitions(choices,0.01,0.01);
        if h
            check(i,2) = 0;
        else
            check(i,2) = 1;
        end
        
        if any(strcmp(uId,uniqueIds{i}))
            check(i,3) = 0;
        else
            check(i,3) = 1;
        end
        
%         % add fourth check
%         if ...
%             mean(choices(1:end-2)==1 & choices(2:end-1)==2 & choices(3:end)==3) > 0.65 || ...
%             mean(choices(1:end-2)==1 & choices(2:end-1)==2 & choices(3:end)==3) < 0.015 || ...
%             mean(choices(1:end-2)==1 & choices(2:end-1)==2 & choices(3:end)==3) > 0.65 || ...
%             mean(choices(1:end-2)==1 & choices(2:end-1)==2 & choices(3:end)==3) < 0.015
%         check(i,4) = 0;
%         else
%             check(i,4) = 1;
%         end
        
        ct = ct+1;
        allmean(ct,:) = [mean(choices==1) mean(choices==2) mean(choices==3)];
        alldevp(ct,1) = devp;
        alldevq(ct,1) = devq;
    else
        check(i,:) = 1;
    end
end
% warning('on','stats:runstest:ValuesOmitted')
allmean(1e5:-1:ct+1,:) = [];
alldevp(ct+1:end,:) = [];
alldevq(ct+1:end,:) = [];
% figure, for i = 1:3, subplot(1,3,i), hist(allmean(:,i),100,0:0.05:1); ylim([0 200]), end

% now remove all HITs of workers where any of the first three checks are 0
% (we aren't doing this)

% ACTUALLY, what we are doing here is removing ANYONE who is beyong 0.4,
% which might be a bit strict considering some people still contribute
% useful information but could also be just fine
rmids = find(check(:,1)==0|check(:,2)==0|check(:,3)==0);
rmind2 = vertcat(allinds{rmids});

% cheaterids = uniqueIds(any(check==0,2));

n_rm = length(rmids);
n_unique = length(uniqueIds);

% and remove all those where RT was too fast (given by rmind)
rmind = unique([rmind;rmind2]);