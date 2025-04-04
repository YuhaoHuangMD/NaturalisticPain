function decodeMomentaryPainWithEphys(subjName, behName, featName,decodeFile,anatName, outputFigDir,groupQuantDir)

load(behName);
load(anatName);
load(decodeFile);
load(featName);

%% DESCRIPTION
% THIS SCRIPT TAKES IN EPHYS FEATURES AND PERFORMS DECODING ON
% MOMENTARY PAIN EVENTS
% The supplied file is the ephys feature data (band power), linked to each
% labeled momentary pain events.
% There are 3 models. The first model uses features from the index
% self-reported pain model. The second model uses a random set of features.
% The third model uses all features and performs elastic regression. 

%% GET FEATURES FROM THE INDEX MODEL PREDICTING SELF-REPORTED PAIN
tfPred = featSave.tfPred;
outcome = featSave.outcome;
rocXTick = [0:0.1:1];
% USE A MODEL TO DECODE SURROUNDING TIME FRAMES
tfPredUse = tfPred';
XTrainComp = tfPredUse;
useSmote = 1;
if useSmote == 1
    [balancedFeat, balancedOutcome] = smote(XTrainComp, [], 'Class', outcome);
    outcomeNew = logical(balancedOutcome);
    XTrainComp = balancedFeat;
else
    outcomeNew = outcome;
end

[B,FitInfo] = lassoglm(XTrainComp,outcomeNew,'binomial','alpha',0.5,'CV',10);
idxLambda = FitInfo.IndexMinDeviance;
B = B(:,idxLambda);
B0 = FitInfo.Intercept(idxLambda);
lambdaUsed = FitInfo.Lambda(idxLambda);
coef = [B0; B];
decodeIdx = find(B);

rng(1,"twister");
outcomeShuffled = outcomeNew(randperm(length(outcomeNew))); % RANDOMLY SHUFFLE
[B,FitInfo] = lassoglm(XTrainComp,outcomeShuffled,'binomial','alpha',0.5,'lambda', lambdaUsed);
B0 = FitInfo.Intercept;
coefShuffle = [B0; B];
shuffleDecodeIdx = find(B);

%% MOMENTARY PAIN DECODING
% set feature names
chanSelect = featSave.chanSel;

% set labels
behDates = behFeatures.linkedPainBeh.blockDate;
behDays = days(behDates - behDates(1));
naturalStatesLabels = behFeatures.linkedPainBeh.blockLabel;
discomfortIdx = contains(naturalStatesLabels, caseInsensitivePattern('Discomfort'));
painIdx = discomfortIdx; 
if strcmp(subjName, 'S20_150')
    neutralIdx = ~discomfortIdx;
else
    neutralIdx = contains(naturalStatesLabels, caseInsensitivePattern('neutral'));
end
outcomeBeh = zeros(length(naturalStatesLabels),1);
outcomeBeh(discomfortIdx) = 1;
outcomeBeh(neutralIdx) = 2;
trialSel = find(outcomeBeh);
outcomeNew = outcomeBeh(trialSel);
outcomeNew(outcomeNew==2) = 0;
behDays = behDays(trialSel);

% GET JUST DISCOMFORT VS NEUTRAL
% POWER
tfpred_natural = behFeatures.behDb(chanSelect,trialSel,:);
tfpred_natural = permute(tfpred_natural, [3,1,2]);
outcome_natural = outcomeNew;
tfpred_natural_reduced = reshape(tfpred_natural,[], size(tfpred_natural,3))';
tfpred_natural_decodeOnly = tfpred_natural_reduced(:,decodeIdx);
outcome = logical(outcome_natural);
outcomeShuffled = logical(outcome_natural(randperm(length(outcome_natural)))); % RANDOMLY SHUFFLE LABEL AND EVALUATE

tfPred = tfpred_natural_decodeOnly;
nboot = 10;
rocXb = [];
rocYb = [];
rocYbsh = [];
rocYb_std = [];
rocYbsh_std = [];
rocAUCb = zeros(nboot,1);
rocAUCb_low = zeros(nboot,1);
rocAUCb_high = zeros(nboot,1);
rocAUCShuffledb = zeros(nboot,1);
rocAUCShuffledb_low = zeros(nboot,1);
rocAUCShuffledb_high = zeros(nboot,1);
accuracyGrp = [];
accuracyGrpShuffle = [];
for b = 1:nboot
    CVFold = 5;
    outcomeCV = cell(CVFold,1);
    outcomeCVShuffled = cell(CVFold,1);
    scoreCV = cell(CVFold,1);
    scoreCVShuffled = cell(CVFold,1);
    accuracy = ones(1, CVFold);
    accuracyShuffle = ones(1, CVFold);

    [epochAssignment,IA,~] = unique(behDays);
    stratOutcomeUse = outcome(IA);
    d = cvpartition(stratOutcomeUse,'KFold',CVFold);

    for n = 1:d.NumTestSets
        
        % split into CV by epochs   
        trainset = epochAssignment(d.training(n));
        testset = epochAssignment(d.test(n));
        
        % get elements of the 1s chunk dataset 
        trainsetidx = zeros(length(behDays),1);
        for x = 1:length(trainset)
            trainsetidx(behDays == trainset(x)) = 1;
        end
        testsetidx = zeros(length(behDays),1);
        for x = 1:length(testset)
            testsetidx(behDays == testset(x)) = 1;
        end
        idxTrain = logical(trainsetidx);
        idxTest = logical(testsetidx);

        XTrain = tfPred(idxTrain,:);
        yTrain = outcome(idxTrain);
        yTrainShuffled = outcomeShuffled(idxTrain);
        XTest = tfPred(idxTest,:);
        yTest = outcome(idxTest);
        %yTestShuffled = outcomeShuffled(idxTest);

        % RESAMPLE TO MATCH TRAINING DATA
        matchSamples = 1;
        if matchSamples == 1
            yPos = find(yTrain==1);
            yPosLength = length(yPos);
            yNeg = find(yTrain==0);
            yNegUse = 1 + (length(yNeg)-1) .* rand(yPosLength,1);
            yNegUse = floor(yNegUse);
            yTrainMatched = [yTrain(yPos); yTrain(yNeg(yNegUse))];
            xTrainMatched = [XTrain(yPos,:); XTrain(yNeg(yNegUse),:)];
            XTrain = xTrainMatched;
            yTrain = yTrainMatched;
            yTrainShuffled = yTrain(randperm(length(yTrain)));  
            
            yPos = find(yTest==1);
            yPosLength = length(yPos);
            yNeg = find(yTest==0);
            yNegUse = 1 + (length(yNeg)-1) .* rand(yPosLength,1); % match with random indices of test data
            yNegUse = floor(yNegUse);
            yTestMatched = [yTest(yPos); yTest(yNeg(yNegUse))];
            xTestMatched = [XTest(yPos,:); XTest(yNeg(yNegUse),:)];

            XTest = xTestMatched;
            yTest = yTestMatched;
        end
        
        Mdl = fitglm(XTrain, yTrain, 'Distribution', 'binomial', 'Link', 'logit');
        [~,post_score_test] = predict(Mdl,XTest);
        yhat = post_score_test(:,2);
        yhatBinom = (yhat>=0.5);
        [tbl, ~, ~, ~] = crosstab(yhatBinom,yTest);
        if size(tbl,1) == 1
            accuracy(n) = tbl(1,1) / length(yTest) * 100;
        elseif size(tbl,2) == 1
            accuracy(n) = tbl(2,1) / length(yTest) * 100;
        else
            accuracy(n) = (tbl(1,1) + tbl(2,2)) / length(yTest) * 100;
        end
        outcomeCV{n} = yTest;
        scoreCV{n} = yhat;

        % SHUFFLED DAT
        Mdl = fitglm(XTrain, yTrainShuffled, 'Distribution', 'binomial', 'Link', 'logit');
        [~,post_score_test] = predict(Mdl,XTest);

        yhat = post_score_test(:,2);
        yhatBinom = (yhat>=0.5);
        [tbl, ~, ~, ~] = crosstab(yhatBinom,yTest);
        if size(tbl,1) == 1
            accuracyShuffle(n) = tbl(1,1) / length(yTest) * 100;
        elseif size(tbl,2) == 1
            accuracyShuffle(n) = tbl(2,1) / length(yTest) * 100;
        else
            accuracyShuffle(n) = (tbl(1,1) + tbl(2,2)) / length(yTest) * 100;
        end
        outcomeCVShuffled{n} = yTest;
        scoreCVShuffled{n} = yhat;
        
    end
    [rocX,rocY,~,rocAUC] = perfcurve(outcomeCV, scoreCV, 'true','XVALS', [0:0.1:1]); %,'XVALS', [0:0.1:1]
    [~,rocYsh,~,rocAUCShuffled] = perfcurve(outcomeCVShuffled, scoreCVShuffled, 'true','XVALS', [0:0.1:1]);
    
    rocXb = cat(2, rocXb, rocX(:,1));
    rocYb = cat(2, rocYb, rocY(:,1));
    rocYbsh = cat(2, rocYbsh, rocYsh(:,1));
    rocYb_std = cat(2, rocYb_std, rocY(:,3)-rocY(:,2));
    rocYbsh_std = cat(2, rocYbsh_std, rocYsh(:,3)-rocYsh(:,2));
    rocAUCb(b) = rocAUC(1);
    rocAUCb_low(b) = rocAUC(2);
    rocAUCb_high(b) = rocAUC(3);
    rocAUCShuffledb(b) = rocAUCShuffled(1);
    rocAUCShuffledb_low(b) = rocAUCShuffled(2);
    rocAUCShuffledb_high(b) = rocAUCShuffled(3);
    accuracyGrp = [accuracyGrp; accuracy'];
    accuracyGrpShuffle = [accuracyGrpShuffle; accuracyShuffle'];
end

figure;
plotSE = mean(rocYb_std,2)./(2*1.96);
toPlot = nanmean(rocYb,2);
ck_shadedErrorBar(nanmean(rocXb,2), toPlot, plotSE,{'color', [22, 160, 133]./255},1); hold on;
plotSE = mean(rocYbsh_std,2)./(2*1.96);
toPlot = nanmean(rocYbsh,2);
ck_shadedErrorBar(nanmean(rocXb,2), toPlot, plotSE,{'color', .5*[1 1 1]},1); 
plot(0:1, 0:1, 'k:');
rocText = cell(2,1);
rocText{1} = [num2str(round(nanmean(rocAUCb),2)) ' (' num2str(round(nanmean(rocAUCb_low),2)) '-' num2str(round(nanmean(rocAUCb_high),2)) ')' ' AUC ' num2str(round(nanmean(accuracyGrp),2)) 'Acc sd+/- ' num2str(round(std(accuracyGrp),2))];
rocText{2} = [num2str(round(nanmean(rocAUCShuffledb),2)) ' (' num2str(round(nanmean(rocAUCShuffledb_low),2)) '-' num2str(round(nanmean(rocAUCShuffledb_high),2)) ')' ' AUC ' num2str(round(nanmean(accuracyGrpShuffle),2)) 'Acc sd+/-' num2str(round(std(accuracyGrpShuffle),2))];
legend({['True ' rocText{1}], ['Shuffled ' rocText{2}]}, 'Location', 'southeast'); legend('boxoff');
ylim([0 1]); xlim([0 1]);
box off; xlabel('False positive rate'); ylabel('True positive rate'); title('Decoding Discomfort Events');

ckSTIM_saveFig(fullfile(outputFigDir,['logistic' '_decoding_discomfort']),10,10,300,'',1,[8,8],[]);
close all;

behSave.rocX = nanmean(rocXb,2);
behSave.rocY = rocYb;
behSave.rocYsh = rocYbsh;
behSave.roc = rocAUCb;
behSave.rocsh = rocAUCShuffledb;

%% RANDOM MODEL FEAT
tfpred_natural_Shuffle = tfpred_natural_reduced(:,shuffleDecodeIdx);

tfPred = tfpred_natural_Shuffle;
nboot = 10;
rocXb = [];
rocYb = [];
rocYbsh = [];
rocYb_std = [];
rocYbsh_std = [];
rocAUCb = zeros(nboot,1);
rocAUCb_low = zeros(nboot,1);
rocAUCb_high = zeros(nboot,1);
rocAUCShuffledb = zeros(nboot,1);
rocAUCShuffledb_low = zeros(nboot,1);
rocAUCShuffledb_high = zeros(nboot,1);
accuracyGrp = [];
accuracyGrpShuffle = [];
for b = 1:nboot
    CVFold = 5;
    outcomeCV = cell(CVFold,1);
    outcomeCVShuffled = cell(CVFold,1);
    scoreCV = cell(CVFold,1);
    scoreCVShuffled = cell(CVFold,1);
    accuracy = ones(1, CVFold);
    accuracyShuffle = ones(1, CVFold);

    [epochAssignment,IA,~] = unique(behDays);
    stratOutcomeUse = outcome(IA);
    d = cvpartition(stratOutcomeUse,'KFold',CVFold);

    for n = 1:d.NumTestSets
        
        % split into CV by epochs   
        trainset = epochAssignment(d.training(n));
        testset = epochAssignment(d.test(n));
        
        % get elements of the 1s chunk dataset 
        trainsetidx = zeros(length(behDays),1);
        for x = 1:length(trainset)
            trainsetidx(behDays == trainset(x)) = 1;
        end
        testsetidx = zeros(length(behDays),1);
        for x = 1:length(testset)
            testsetidx(behDays == testset(x)) = 1;
        end
        idxTrain = logical(trainsetidx);
        idxTest = logical(testsetidx);

        XTrain = tfPred(idxTrain,:);
        yTrain = outcome(idxTrain);
        yTrainShuffled = outcomeShuffled(idxTrain);
        XTest = tfPred(idxTest,:);
        yTest = outcome(idxTest);

        % RESAMPLE TO MATCH TRAINING DATA
        matchSamples = 1;
        if matchSamples == 1
            yPos = find(yTrain==1);
            yPosLength = length(yPos);
            yNeg = find(yTrain==0);
            yNegUse = 1 + (length(yNeg)-1) .* rand(yPosLength,1);
            yNegUse = floor(yNegUse);
            yTrainMatched = [yTrain(yPos); yTrain(yNeg(yNegUse))];
            xTrainMatched = [XTrain(yPos,:); XTrain(yNeg(yNegUse),:)];
            XTrain = xTrainMatched;
            yTrain = yTrainMatched;
            yTrainShuffled = yTrain(randperm(length(yTrain)));  
            
            yPos = find(yTest==1);
            yPosLength = length(yPos);
            yNeg = find(yTest==0);
            yNegUse = 1 + (length(yNeg)-1) .* rand(yPosLength,1); % match with random indices of test data
            yNegUse = floor(yNegUse);
            yTestMatched = [yTest(yPos); yTest(yNeg(yNegUse))];
            xTestMatched = [XTest(yPos,:); XTest(yNeg(yNegUse),:)];

            XTest = xTestMatched;
            yTest = yTestMatched;
        end

        Mdl = fitglm(XTrain, yTrain, 'Distribution', 'binomial', 'Link', 'logit');
        [~,post_score_test] = predict(Mdl,XTest);
        yhat = post_score_test(:,2);
        yhatBinom = (yhat>=0.5);
        [tbl, ~, ~, ~] = crosstab(yhatBinom,yTest);
        if size(tbl,1) == 1
            accuracy(n) = tbl(1,1) / length(yTest) * 100;
        elseif size(tbl,2) == 1
            accuracy(n) = tbl(2,1) / length(yTest) * 100;
        else
            accuracy(n) = (tbl(1,1) + tbl(2,2)) / length(yTest) * 100;
        end
        outcomeCV{n} = yTest;
        scoreCV{n} = yhat;

        % SHUFFLED DAT
        Mdl = fitglm(XTrain, yTrainShuffled, 'Distribution', 'binomial', 'Link', 'logit');
        [~,post_score_test] = predict(Mdl,XTest);
        yhat = post_score_test(:,2);
        yhatBinom = (yhat>=0.5);
        [tbl, ~, ~, ~] = crosstab(yhatBinom,yTest);
        if size(tbl,1) == 1
            accuracyShuffle(n) = tbl(1,1) / length(yTest) * 100;
        elseif size(tbl,2) == 1
            accuracyShuffle(n) = tbl(2,1) / length(yTest) * 100;
        else
            accuracyShuffle(n) = (tbl(1,1) + tbl(2,2)) / length(yTest) * 100;
        end
        outcomeCVShuffled{n} = yTest;
        scoreCVShuffled{n} = yhat;
        
    end
    [rocX,rocY,~,rocAUC] = perfcurve(outcomeCV, scoreCV, 'true','XVALS', [0:0.1:1]); %,'XVALS', [0:0.1:1]
    [~,rocYsh,~,rocAUCShuffled] = perfcurve(outcomeCVShuffled, scoreCVShuffled, 'true','XVALS', [0:0.1:1]);
    
    rocXb = cat(2, rocXb, rocX(:,1));
    rocYb = cat(2, rocYb, rocY(:,1));
    rocYbsh = cat(2, rocYbsh, rocYsh(:,1));
    rocYb_std = cat(2, rocYb_std, rocY(:,3)-rocY(:,2));
    rocYbsh_std = cat(2, rocYbsh_std, rocYsh(:,3)-rocYsh(:,2));
    rocAUCb(b) = rocAUC(1);
    rocAUCb_low(b) = rocAUC(2);
    rocAUCb_high(b) = rocAUC(3);
    rocAUCShuffledb(b) = rocAUCShuffled(1);
    rocAUCShuffledb_low(b) = rocAUCShuffled(2);
    rocAUCShuffledb_high(b) = rocAUCShuffled(3);
    accuracyGrp = [accuracyGrp; accuracy'];
    accuracyGrpShuffle = [accuracyGrpShuffle; accuracyShuffle'];
end

figure;
plotSE = mean(rocYb_std,2)./(2*1.96);
toPlot = nanmean(rocYb,2);
ck_shadedErrorBar(nanmean(rocXb,2), toPlot, plotSE,{'color', [22, 160, 133]./255},1); hold on;
plotSE = mean(rocYbsh_std,2)./(2*1.96);
toPlot = nanmean(rocYbsh,2);
ck_shadedErrorBar(nanmean(rocXb,2), toPlot, plotSE,{'color', .5*[1 1 1]},1); 
plot(0:1, 0:1, 'k:');
rocText = cell(2,1);
rocText{1} = [num2str(round(nanmean(rocAUCb),2)) ' (' num2str(round(nanmean(rocAUCb_low),2)) '-' num2str(round(nanmean(rocAUCb_high),2)) ')' ' AUC ' num2str(round(nanmean(accuracyGrp),2)) 'Acc sd+/- ' num2str(round(std(accuracyGrp),2))];
rocText{2} = [num2str(round(nanmean(rocAUCShuffledb),2)) ' (' num2str(round(nanmean(rocAUCShuffledb_low),2)) '-' num2str(round(nanmean(rocAUCShuffledb_high),2)) ')' ' AUC ' num2str(round(nanmean(accuracyGrpShuffle),2)) 'Acc sd+/-' num2str(round(std(accuracyGrpShuffle),2))];
legend({['True ' rocText{1}], ['Shuffled ' rocText{2}]}, 'Location', 'southeast'); legend('boxoff');
ylim([0 1]); xlim([0 1]);
box off; xlabel('False positive rate'); ylabel('True positive rate'); title('Decoding Discomfort Events');

ckSTIM_saveFig(fullfile(outputFigDir,['logistic' '_decoding_discomfort_rand']),10,10,300,'',1,[8,8],[]);
close all;

behSave.rocX_rand = rocXb;
behSave.rocY_rand = rocYb;
behSave.rocYsh_rand = rocYbsh;
behSave.roc_rand = rocAUCb;
behSave.rocsh_rand = rocAUCShuffledb;

%% POWER ANALYSIS 
behSave.discomfortOutcome = outcomeNew;
behSave.tfpred = tfpred_natural_reduced;
behSave.decodeIdx = decodeIdx;
behSave.shuffleIdx = shuffleDecodeIdx;
behSave.featSave = featSave;

disp([subjName ' the auc for decoder feat is: ' num2str(mean(behSave.roc)) ' for rand pred is: ' num2str(mean(behSave.roc_rand))])


%% DECODING USING ALL DATA

% set feature names
chanSelect = featSave.chanSel;
powFeaturesVec = featSave.powFeaturesVec;

% set labels
behDates = behFeatures.linkedPainBeh.blockDate;
behDays = days(behDates - behDates(1));
naturalStatesLabels = behFeatures.linkedPainBeh.blockLabel;
discomfortIdx = contains(naturalStatesLabels, caseInsensitivePattern('Discomfort'));
painIdx = discomfortIdx; 
if strcmp(subjName, 'S20_150')
    neutralIdx = ~discomfortIdx;
else
    neutralIdx = contains(naturalStatesLabels, caseInsensitivePattern('neutral'));
end
outcomeBeh = zeros(length(naturalStatesLabels),1);
outcomeBeh(discomfortIdx) = 1;
outcomeBeh(neutralIdx) = 2;
trialSel = find(outcomeBeh);
outcomeNew = outcomeBeh(trialSel);
outcomeNew(outcomeNew==2) = 0;
behDays = behDays(trialSel);

% GET JUST DISCOMFORT VS NEUTRAL
tfpred_natural = behFeatures.behDb(chanSelect,trialSel,:);
tfpred_natural = permute(tfpred_natural, [3,1,2]);
outcome_natural = outcomeNew;
tfpred_natural_reduced = reshape(tfpred_natural,[], size(tfpred_natural,3))';
outcome = logical(outcome_natural);
outcomeShuffled = logical(outcome_natural(randperm(length(outcome_natural)))); % RANDOMLY SHUFFLE LABEL AND EVALUATE
tfPred = tfpred_natural_reduced;

nboot = 10;
CVFold = 5;
rocXTick = [0:0.1:1];
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

    [epochAssignment,IA,~] = unique(behDays);
    stratOutcomeUse = outcome(IA);
    d = cvpartition(stratOutcomeUse,'KFold',CVFold);

    for n = 1:CVFold
        % split into CV by epochs   
        trainset = epochAssignment(d.training(n));
        testset = epochAssignment(d.test(n));
        
        % get elements of the 1s chunk dataset 
        trainsetidx = zeros(length(behDays),1);
        for x = 1:length(trainset)
            trainsetidx(behDays == trainset(x)) = 1;
        end
        testsetidx = zeros(length(behDays),1);
        for x = 1:length(testset)
            testsetidx(behDays == testset(x)) = 1;
        end
        idxTrain = logical(trainsetidx);
        idxTest = logical(testsetidx);
        XTrain = tfPred(idxTrain,:);
        yTrain = outcome(idxTrain);
        yTrainShuffled = outcomeShuffled(idxTrain);
        XTest = tfPred(idxTest,:);
        yTest = outcome(idxTest);

        matchSamples = 1;
        if matchSamples == 1
            yPos = find(yTrain==1);
            yPosLength = length(yPos);
            yNeg = find(yTrain==0);
            yNegUse = 1 + (length(yNeg)-1) .* rand(yPosLength,1);
            yNegUse = floor(yNegUse);
            yTrainMatched = [yTrain(yPos); yTrain(yNeg(yNegUse))];
            xTrainMatched = [XTrain(yPos,:); XTrain(yNeg(yNegUse),:)];
            XTrain = xTrainMatched;
            yTrain = yTrainMatched;
            yTrainShuffled = yTrain(randperm(length(yTrain)));  
            
            yPos = find(yTest==1);
            yPosLength = length(yPos);
            yNeg = find(yTest==0);
            yNegUse = 1 + (length(yNeg)-1) .* rand(yPosLength,1); % match with random indices of test data
            yNegUse = floor(yNegUse);
            yTestMatched = [yTest(yPos); yTest(yNeg(yNegUse))];
            xTestMatched = [XTest(yPos,:); XTest(yNeg(yNegUse),:)];

            XTest = xTestMatched;
            yTest = yTestMatched;
        end

        [B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','alpha',0.5, 'CV',10);
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

disp(['total var auc for ' subjName ' ' num2str(mean(meanAUC_save))]);


behSave.meanAUC = meanAUC_save;
behSave.meanAUC_shuffled = meanAUC_shuffled_save;
behSave.meanAUCUpper = meanAUCUpper;
behSave.meanAUCLower = meanAUCLower;
behSave.ROCX = ROCX_save;
behSave.ROCY = ROCY_save;
behSave.ROCY_shuffled = ROCY_shuffled_save;
behSave.accuracyGrp = accuracyGrp;
behSave.accuracyGrpShuffle = accuracyGrpShuffle;
behSave.powFeaturesVec = powFeaturesVec;
behSave.tfPred = tfPred;
behSave.outcome = outcome;
behSave.accuracyGrp = accReshape;
behSave.accuracyGrpShuffle = accShuffleReshape;
behSave.featImp = featImpCollapsed; 
behSave.featShuffleImp = featImpShuffleCollapsed; 

save(fullfile(groupQuantDir,[subjName 'beh-AUC']), 'behSave', '-v7.3');

