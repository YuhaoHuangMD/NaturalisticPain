function decodeMomentaryPainWithCV(subjName, fileName,groupQuantDir)

%% DESCRIPTION
% THIS SCRIPT TAKES IN FACIAL FEATURES AND PERFORMS DECODING ON
% MOMENTARY PAIN EVENTS
% The needed file is the facial feature data, which contains AU features for
% each momentary pain episode. 

load(fileName);

faceDat = linkedFaceBeh.blockEpoch;
faceDatMean = nanmean(faceDat,3);

% SET OUTCOMES
behDates = linkedFaceBeh.blockDate;
behDays = days(behDates - behDates(1));
naturalStatesLabels = linkedFaceBeh.blockLabel;
discomfortIdx = contains(naturalStatesLabels, caseInsensitivePattern('Discomfort'));
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
auFeat = faceDatMean(:,trialSel);
outcome = logical(outcomeNew);
outcomeShuffled = logical(outcomeNew(randperm(length(outcomeNew)))); % RANDOMLY SHUFFLE LABEL AND EVALUATE

tfPred = auFeat';
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
        
        lambdaUsed = 0.1;
        [B,FitInfo] = lassoglm(XTrain,yTrain,'binomial','alpha',0.5, 'lambda',lambdaUsed);
        B0 = FitInfo.Intercept;
        coef = [B0; B];
        yhat = glmval(coef,XTest,'logit');
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
        [B,FitInfo] = lassoglm(XTrain,yTrainShuffled,'binomial','alpha',0.5, 'lambda',lambdaUsed);
        B0 = FitInfo.Intercept;
        coef = [B0; B];
        yhat = glmval(coef,XTest,'logit');
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

cvDecodeBeh = [];
cvDecodeBeh.auc = rocAUCb;
cvDecodeBeh.aucsh = rocAUCShuffledb;
cvDecodeBeh.rocXgroup = mean(rocXb,2);
cvDecodeBeh.rocYgroup = rocYb;
cvDecodeBeh.rocYgroupsh = rocYbsh;

save(fullfile(groupQuantDir,[subjName 'decode-cv-beh']), 'cvDecodeBeh', '-v7.3');
