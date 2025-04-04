function decodePainWithCV(subjName, cvfeatFile, groupQuantDir)

load(cvfeatFile);

%% DESCRIPTION
% THIS SCRIPT TAKES IN FACIAL FEATURES AND PERFORMS DECODING ON
% SELF-REPORTED PAIN SCORES
% cvfeatFile is the facial feature data, which contains AU features for
% each self-reported pain events. 

outcome = cvFeat.pain >= median(cvFeat.pain);
rocXgroup = [];
rocYgroup = [];
rocYgroupsh = [];
auEvent = [];
auDuration = [];

auLabel = {'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11', ...
    'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20', ...
    'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39'};

%% MODEL DECODE
t = 47; % THIS IS the index for 5min right prior to pain report. 
auFeat1 = squeeze(cvFeat.auDur(:,t,:))';
auFeat2 = squeeze(cvFeat.auEvents(:,t,:))';
auFeat = cat(1, normalize(auFeat1,2), normalize(auFeat2,2));
auFeat(isnan(auFeat)) = 0;
nbootstrap = 100;
aucCollect = zeros(nbootstrap,2);
aucCI = zeros(nbootstrap,2);
for c = 1:nbootstrap
    CVFold = 10;
    outcomeCV = cell(CVFold,1);
    outcomeCVShuffled = cell(CVFold,1);
    scoreCV = cell(CVFold,1);
    scoreCVShuffled = cell(CVFold,1);
    d = cvpartition(outcome,'KFold',CVFold); % Cross-validated ROC

    for n = 1:d.NumTestSets
        idxTrain = d.training(n);
        idxTest = d.test(n);
        
        XTrain = auFeat(:,idxTrain)';
        yTrain = outcome(idxTrain);

        XTest = auFeat(:,idxTest)';
        yTest = outcome(idxTest);

        yTrainShuffled = yTrain(randperm(length(yTrain))); % RANDOMLY SHUFFLE

        [B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','alpha',0.5, 'CV',10);
        idxLambda = FitInfo.IndexMinDeviance;
        B = B(:,idxLambda);
        B0 = FitInfo.Intercept(idxLambda);
        lambdaUsed = FitInfo.Lambda(idxLambda);
        coef = [B0; B];
        yhat = glmval(coef,XTest,'logit');

        outcomeCV{n} = yTest;
        scoreCV{n} = yhat;

        % SHUFFLED DAT
        [B,FitInfo] = lassoglm(XTrain,yTrainShuffled,'binomial','alpha',0.5, 'lambda',lambdaUsed);
        B0 = FitInfo.Intercept;
        coef = [B0; B];
        yhat = glmval(coef,XTest,'logit');
        outcomeCVShuffled{n} = yTest;
        scoreCVShuffled{n} = yhat;
    end
    [rocX,rocY,~,rocAUC] = perfcurve(outcomeCV, scoreCV, 'true','XVALS', [0:0.1:1]);
    [~,rocYsh,~,rocAUCShuffled] = perfcurve(outcomeCVShuffled, scoreCVShuffled, 'true','XVALS', [0:0.1:1]);
    
    aucCollect(c,1) = rocAUC(1);
    aucCollect(c,2) = rocAUCShuffled(1);

    aucCI(c,1) = rocAUC(2);
    aucCI(c,2) = rocAUC(3);

    rocXgroup = cat(3,rocXgroup, rocX);
    rocYgroup = cat(3,rocYgroup, rocY);
    rocYgroupsh = cat(3,rocYgroupsh, rocYsh);

    auEvent = cat(3, auEvent, auFeat2);
    auDuration = cat(3, auDuration, auFeat1);
end

cvDecode = [];
cvDecode.aucCollect = aucCollect;
cvDecode.aucCI = aucCI;
cvDecode.rocXgroup = rocXgroup;
cvDecode.rocYgroup = rocYgroup;
cvDecode.rocYgroupsh = rocYgroupsh;
cvDecode.time = cvFeat.binTime;
cvDecode.auEvent = auEvent;
cvDecode.auDuration = auDuration;
cvDecode.outcome = outcome;
cvDecode.auLabel = auLabel;

disp([subjName ' face auc ' num2str(mean(aucCollect(:,1))) ' ' num2str(mean(aucCI(:,1))) ' - ' num2str(mean(aucCI(:,2)))]);

%% TRAIN ON INDEX TIME USING THE FULL DATA AND DO INFERENCE ON OTHER TIMES
optimalIdx = 47;
auFeat1 = squeeze(cvFeat.auDur(:,optimalIdx,:))';
auFeat2 = squeeze(cvFeat.auEvents(:,optimalIdx,:))';
auFeat = cat(1, auFeat1, auFeat2);

XTrain = auFeat';
yTrain = outcome;
[B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','alpha',0.5, 'CV',10);
idxLambda = FitInfo.IndexMinDeviance;
B = B(:,idxLambda);
B0 = FitInfo.Intercept(idxLambda);
lambdaUsed = FitInfo.Lambda(idxLambda);
coef = [B0; B];

nbootstrap = 100;
coefShuffle = zeros(size(XTrain,2)+1, nbootstrap);
for n = 1:nbootstrap
    outcomeShuffled = outcome(randperm(length(outcome))); % RANDOMLY SHUFFLE
    [B,FitInfo] = lassoglm(XTrain,outcomeShuffled,'binomial','lambda', lambdaUsed);
    B0 = FitInfo.Intercept;
    coefShuffle(:,n) = [B0; B];
end

otherProb = zeros(size(cvFeat.binTime,2),length(outcome));
otherAUC = zeros(size(cvFeat.binTime,2),1);
otherAUCsh = zeros(size(cvFeat.binTime,2),nbootstrap);
otherProbsh = zeros(size(cvFeat.binTime,2),length(outcome));
rocXTick = [0:0.1:1];
for t = 1:size(cvFeat.binTime,2)-1
    auFeat1 = squeeze(cvFeat.auDur(:,t,:))';
    auFeat2 = squeeze(cvFeat.auEvents(:,t,:))';
    auFeat = cat(1, auFeat1, auFeat2);
    XTestComp = auFeat';
    yhat = glmval(coef,XTestComp,'logit');

    otherProb(t,:) = yhat;
    yhatBinom = (yhat>=0.5);
    [tbl, ~, ~, ~] = crosstab(yhatBinom,outcome);
    [rocX,rocY,~,rocAUC] = perfcurve(outcome, yhat, 'true','XVALS', rocXTick); %,'XVALS', [0:0.1:1]
    otherAUC(t) = rocAUC;

    yhatshboot = zeros(nbootstrap,length(outcome));
    for n = 1:nbootstrap
        yhatsh = glmval(coefShuffle(:,n),XTestComp,'logit');
        [~,~,~,rocAUC] = perfcurve(outcome, yhatsh, 'true','XVALS', rocXTick); %,'XVALS', [0:0.1:1]
        yhatshboot(n,:) = yhatsh;
        otherAUCsh(t,n) = rocAUC;
    end
    otherProbsh(t,:) = nanmean(yhatshboot,1);
end

cvDecode.otherProbsh= otherProbsh;
cvDecode.otherAUC =otherAUC;
cvDecode.otherProb = otherProb;
cvDecode.otherAUCsh = otherAUCsh;

save(fullfile(groupQuantDir,[subjName 'decode-cv']), 'cvDecode', '-v7.3');
