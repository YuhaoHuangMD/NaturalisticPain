function decodePainWithEphys(subjName, fileName,anatName, outputFigDir,groupQuantDir)

load(fileName);
load(anatName);

%% DESCRIPTION
% THIS SCRIPT TAKES IN EPHYS FEATURES AND PERFORMS DECODING ON
% SELF-REPORTED PAIN SCORES
% The supplied file is the ephys feature data (band power), linked to each
% self-reported pain event. 

%% GET FEATURES FOR PAIN STATES DECODING
structDat = featuresData;
numTimBin = length(structDat.featBin);
elecInfo = elecRevLook;

timBinSel = numTimBin; % index for 5minutes prior to report of pain
featDat = structDat.featBin{timBinSel};
rejectOut = 1; % reject trial based outliers
useSparsePower = 1; % whether to use non-sparse or sparse data (Sparse data is averaged over 5min whereas non sparse uses 10s windows)

% invariants
timBin = featDat.binTime;
chanSel = 1:size(featDat.tfOrig,2);
trialSel = 1:size(featDat.tfOrig,3);
chanLabels = elecInfo.elec;
bandPower = {'d', 't', 'a', 'b', 'g', 'hg'};
bands = length(bandPower);

% using sparse data or not
if useSparsePower == 1 % use 5min windows
    power = featDat.tfOrig;
else % use 10s windows
    power = featDat.tf;
end

% artifact rejection based on threshold
trialOutlier = []; channelOutlier = [];
if rejectOut == 1
    artifactThresholdStd = 10;
    if size(power,4) ~= 1
        powerOutlier = nanmean(power,4);
    else
        powerOutlier = power;
    end

    trialAvg = squeeze(mean(mean(powerOutlier)));
    channelAvg = mean(squeeze(mean(powerOutlier,1)),2);
    [~, xiOut] = hampel(trialAvg, length(trialAvg), artifactThresholdStd);
    trialOutlier = find(xiOut);
    [~, xiOut] = hampel(channelAvg, length(channelAvg), artifactThresholdStd);
    channelOutlier = find(xiOut);

    trialSel(trialOutlier) = [];
    
    if ~isempty(trialOutlier)
        disp(['rejecting trial ' num2str(trialOutlier')]);
    end
    if ~isempty(channelOutlier)
        disp(['rejecting channels ' num2str(channelOutlier')]);
    end
end
% remove unknowns and uncommon channels from channel set
unknownChanIdx = any([contains(elecInfo.location, 'Unknown'), contains(elecInfo.location, 'lingual'), contains(elecInfo.location, 'occipital'), contains(elecInfo.location, 'calcarine'), contains(elecInfo.location, 'cuneus')],2);
unknownChanIdx = find(unknownChanIdx);

% get new channel set
chanSel([unknownChanIdx,channelOutlier]) = [];
chanNum = length(chanSel);
chanLabels = chanLabels(chanSel);

% get new dataset
if size(power,4) ~= 1
    powerFeat = power(:,chanSel,trialSel,:);
    powerFeat = reshape(powerFeat,[], size(powerFeat,3),size(powerFeat,4));
else
    powerFeat = power(:,chanSel,trialSel);
    powerFeat = reshape(powerFeat,[], size(powerFeat,3));
end

%outcomes dataset
outcome = structDat.painContinuous' >= median(structDat.painContinuous);
outcome = outcome(trialSel);

% power labels
powFeatures = cell(bands, chanNum);
for p = 1:bands
    for c = 1:chanNum
        powFeatures{p,c} = [chanLabels{c},' ',bandPower{p}];
    end
end
powFeaturesVec = reshape(powFeatures,[],1);

% regularized L1 regression or elastic net
tfPred = powerFeat;

rocXTick = [0:0.1:1];
% features tracker
if length(outcome)<50
    nboot = 20; 
    CVFold = 5;
else
    nboot = 10; 
    CVFold = 10;
end
featCoef = zeros(nboot,CVFold, length(powFeaturesVec));
featCoefShuffle = zeros(nboot,CVFold, length(powFeaturesVec));
rocXb = zeros(nboot, length(rocXTick));
rocYb = zeros(nboot, length(rocXTick));
rocYb_std = zeros(nboot, length(rocXTick));
rocYbsh_std = zeros(nboot, length(rocXTick));
rocYbsh = zeros(nboot, length(rocXTick));
rocAUCb = zeros(nboot,1);
rocAUCb_low = zeros(nboot,1);
rocAUCb_high = zeros(nboot,1);
rocAUCShuffledb = zeros(nboot,1);
rocAUCShuffledb_low = zeros(nboot,1);
rocAUCShuffledb_high = zeros(nboot,1);
accuracyGrp = zeros(nboot, CVFold);
accuracyGrpShuffle = zeros(nboot, CVFold);
parfor nr = 1:nboot
    
    outcomeCV = cell(CVFold,1);
    outcomeCVShuffled = cell(CVFold,1);
    scoreCV = cell(CVFold,1);
    scoreCVShuffled = cell(CVFold,1);
    d = cvpartition(outcome,'KFold',CVFold); % Cross-validated ROC

    for n = 1:CVFold
        idxTrain = d.training(n);
        idxTest = d.test(n);
        
        if size(tfPred,3) ~= 1
            XTrain = reshape(tfPred(:,idxTrain,:), size(tfPred,1), [])';
            yTrainTemp = repmat(outcome(idxTrain),[1,size(tfPred,3)]);
            yTrain = reshape(yTrainTemp, [],1);

            XTest = reshape(tfPred(:,idxTest,:), size(tfPred,1), [])';
            yTestTemp = repmat(outcome(idxTest),[1,size(tfPred,3)]);
            yTest = reshape(yTestTemp, [],1);
        else
            XTrain = tfPred(:,idxTrain)';
            yTrain = outcome(idxTrain);

            XTest = tfPred(:,idxTest)';
            yTest = outcome(idxTest);
        end
        % ADDRESS CLASS IMBALANCE WITH SMOTE
        useSmote = 1;
        if useSmote == 1
            [balancedFeat, balancedOutcome] = smote(XTrain, [], 'Class', yTrain);
            yTrain = balancedOutcome;
            XTrain = balancedFeat;
        end
        yTrainShuffled = yTrain(randperm(length(yTrain))); % RANDOMLY SHUFFLE

        [B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','alpha', 0.5,'CV',10);
        idxLambda = FitInfo.IndexMinDeviance;
        B = B(:,idxLambda);
        B0 = FitInfo.Intercept(idxLambda);
        lambdaUsed = FitInfo.Lambda(idxLambda);
        coef = [B0; B];
        yhat = glmval(coef,XTest,'logit');
        featCoef(nr,n,:) = B;
        
        % prediction
        yhatBinom = (yhat>=0.5);
        [tbl, ~, ~, ~] = crosstab(yhatBinom,yTest);
        if size(tbl,1) == 1
            accuracy = tbl(1,1) / length(yTest) * 100;
        elseif size(tbl,2) == 1
            accuracy = tbl(2,1) / length(yTest) * 100;
        else
            accuracy = (tbl(1,1) + tbl(2,2)) / length(yTest) * 100;
        end
        outcomeCV{n} = yTest;
        scoreCV{n} = yhat;
        accuracyGrp(nr,n) = accuracy;

        % SHUFFLED DAT
        [B,FitInfo] = lassoglm(XTrain,yTrainShuffled,'binomial','alpha', 0.5, 'lambda', lambdaUsed);         
        B0 = FitInfo.Intercept;
        coef = [B0; B];
        yhat = glmval(coef,XTest,'logit');
        featCoefShuffle(nr,n,:) = B;

        yhatBinom = (yhat>=0.5);
        [tbl, ~, ~, ~] = crosstab(yhatBinom,yTest);
        if size(tbl,1) == 1
            accuracyShuffle = tbl(1,1) / length(yTest) * 100;
        elseif size(tbl,2) == 1
            accuracyShuffle = tbl(2,1) / length(yTest) * 100;
        else
            accuracyShuffle = (tbl(1,1) + tbl(2,2)) / length(yTest) * 100;
        end
        outcomeCVShuffled{n} = yTest;
        scoreCVShuffled{n} = yhat;
        accuracyGrpShuffle(nr,n) = accuracyShuffle;
    end
    [rocX,rocY,~,rocAUC] = perfcurve(outcomeCV, scoreCV, 'true', 'XVALS',rocXTick);
    rocY(1,:) = 0;
    [~,rocYsh,~,rocAUCShuffled] = perfcurve(outcomeCVShuffled, scoreCVShuffled, 'true','XVALS', rocXTick);
    rocYsh(1,:) = 0;

    rocXb(nr,:) = rocX(:,1);
    rocYb(nr,:) = rocY(:,1);
    rocYbsh_std(nr,:) = rocYsh(:,3)-rocYsh(:,2);
    rocYbsh(nr,:) = rocYsh(:,1);
    rocYb_std(nr,:) = rocY(:,3)-rocY(:,2);
    rocAUCb(nr) = rocAUC(1);
    rocAUCb_low(nr) = rocAUC(2);
    rocAUCb_high(nr) = rocAUC(3);
    rocAUCShuffledb(nr) = rocAUCShuffled(1);
    rocAUCShuffledb_low(nr) = rocAUCShuffled(2);
    rocAUCShuffledb_high(nr) = rocAUCShuffled(3);
end

% VAR COLLECT
meanAUC_save = rocAUCb;
meanAUC_shuffled_save = rocAUCShuffledb;
meanAUCUpper = rocAUCb_high;
meanAUCLower = rocAUCb_low;
ROCX_save = nanmean(rocXb,1);
ROCY_save = rocYb;
ROCY_shuffled_save = rocYbsh;
featImpCollapsed = reshape(featCoef, [],size(featCoef,3));
featImpShuffleCollapsed = reshape(featCoefShuffle, [],size(featCoefShuffle,3));
accReshape = reshape(accuracyGrp, [],1);
accShuffleReshape = reshape(accuracyGrpShuffle, [],1);

disp(['auc for ' subjName ' ' num2str(mean(meanAUC_save))]);


figure;
plotSE = mean(rocYb_std,1)./(2*1.96);
toPlot = nanmean(rocYb,1);
ck_shadedErrorBar(nanmean(rocXb,1), toPlot, plotSE,{'color', [22, 160, 133]./255},1); hold on;
plotSE = mean(rocYbsh_std,1)./(2*1.96);
toPlot = nanmean(rocYbsh,1);
ck_shadedErrorBar(nanmean(rocXb,1), toPlot, plotSE,{'color', .5*[1 1 1]},1); 
plot(0:1, 0:1, 'k:');
rocText = cell(2,1);
rocText{1} = [num2str(round(nanmean(rocAUCb),1)) ' (' num2str(round(nanmean(rocAUCb_low),1)) '-' num2str(round(nanmean(rocAUCb_high),1)) ')' ' AUC ' num2str(round(nanmean(accReshape),1)) 'Acc sd+/- ' num2str(round(std(accReshape),1))];
rocText{2} = [num2str(round(nanmean(rocAUCShuffledb),1)) ' (' num2str(round(nanmean(rocAUCShuffledb_low),1)) '-' num2str(round(nanmean(rocAUCShuffledb_high),1)) ')' ' AUC ' num2str(round(nanmean(accShuffleReshape),1)) 'Acc sd+/-' num2str(round(std(accShuffleReshape),1))];
legend({['True ' rocText{1}], ['Shuffled ' rocText{2}]}, 'Location', 'southeast'); legend('boxoff');
ylim([0 1]); xlim([0 1]);
box off; xlabel('False positive rate'); ylabel('True positive rate'); title('Decoding Pain Self-Reports');

ckSTIM_saveFig(fullfile(outputFigDir,['elastic' '_decoding' '_timBin_' num2str(timBin) '_' num2str(timBinSel) ]),10,10,300,'',1,[12,10],[]);
close all;

% ACCURACY PLOT
x1 = accReshape;
x2 = accShuffleReshape;
[~,p] = ttest(x1, x2);
al_goodplot(x1,1,0.5,[22, 160, 133]/255,'left',[],[],0.2);
al_goodplot(x2,1,0.5,.5*[1 1 1],'right',[],[],0.2);
text(1,max([x1; x2]),['p: ' num2str(p)]);
box off; ylabel('Accuracy');
legend('Decoder','Shuffled'); legend('boxoff');
ckSTIM_saveFig(fullfile(outputFigDir,['elastic_accuracy' '_decoding' '_timBin_' num2str(timBin) '_' num2str(timBinSel) ]),10,10,300,'',1,[7,10],[]);
close all;

% COLLATE VARIABLES FROM DECODER
featSave = [];
featSave.chanLabels = chanLabels; 
featSave.chanSel = chanSel;
featSave.meanAUC = meanAUC_save;
featSave.meanAUC_shuffled = meanAUC_shuffled_save;
featSave.meanAUCUpper = meanAUCUpper;
featSave.meanAUCLower = meanAUCLower;
featSave.ROCX = ROCX_save;
featSave.ROCY = ROCY_save;
featSave.ROCY_shuffled = ROCY_shuffled_save;
featSave.accuracyGrp = accuracyGrp;
featSave.accuracyGrpShuffle = accuracyGrpShuffle;
featSave.chanLocation = elecInfo.location;
featSave.chanNativeCoords = elecInfo.elecpos;
featSave.chanMNICoords = elecInfo.avgCoords;
featSave.powFeaturesVec = powFeaturesVec;
featSave.tfPred = tfPred;
featSave.outcome = outcome;
featSave.outcomeContinuous = structDat.painContinuous(trialSel);
featSave.accuracyGrp = accReshape;
featSave.accuracyGrpShuffle = accShuffleReshape;
featSave.featImp = featImpCollapsed; 
featSave.featShuffleImp = featImpShuffleCollapsed; 

save(fullfile(groupQuantDir,[subjName 'decode-AUC-elastic']), 'featSave', '-v7.3');

