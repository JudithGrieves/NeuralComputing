%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Judith Grieves - Neural Computing Coursework - March 2020
% Train an SVM classifier
%
% Auto generated code - initially from MATLAB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("### Running SVMtrainClassifier.m");

clear all;
close all; 
OverallResults=[]; % initialise a matrix to hold the grid search results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read in the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
InputFile="Train-breast-cancer-coded.csv"; 
disp("Input File: " + InputFile);
trainingData = readtable(InputFile); % read in the training set

% to test generalisation in the grid search
InputFile="Test-breast-cancer-coded.csv";
disp("Input File: " + InputFile);
testData = readtable(InputFile); % read in the test set 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data transformation: Select subset of the features
% features were added in this order: malig, caps, irrad, size, age
% Final selection: malig, caps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predictorNames = {'class', 'age', 'menopausal', 'tumorSize', 'nodes', 'nodeCaps', 'malignancy', 'leftRight', 'quadrant', 'irradiated'};
includedPredictorNames = trainingData.Properties.VariableNames([false false false false false true true false false false]); % run for various values     
trainX = trainingData(:,includedPredictorNames);
trainT = trainingData.class;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run grid search function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Results] = RunGridSearch(trainX,trainT,testData); 
% add the trial training ratio to the results grid
OverallResults = [OverallResults; zeros(size(Results,1),1) Results];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% write all the results to a CSV file for evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputFile= 'SVMParamsAccuracy.csv';
writematrix(OverallResults,outputFile); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  select best model hyperparameters and train a final model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accTrain= OverallResults(:,6); % Training accuracy
accTest =  OverallResults(:,12); % Test accuracy
totalTrain = size(trainX,1); % number of training instances
totalTest = size(testData,1); % number of test instances

% find the results row with the highest overall train/test accuracy
accMean =((str2double(accTrain)  * totalTrain) + (str2double(accTest) * totalTest)) / (totalTrain + totalTest); % overall mean accuracy of test/train
maxVal = max(accMean);
BestResults = OverallResults(accMean== max(accMean),:); % result rows with the highest overall accuracy

% build the final model from the best parameter combination
bestKernel=BestResults(1,3); 
bestBox=str2double(BestResults(1,4));
bestOrder=str2double(BestResults(1,5));
disp("creating best SVM model for best overall accuracy: " +  num2str(maxVal) + " : " + bestKernel+ " : "  + bestBox + " : " + bestOrder );

if strcmp(bestKernel,'polynomial') == 1
    SVMtrainedClassifier = fitcsvm(trainX,trainT,'Standardize',true,'KernelFunction',bestKernel,'BoxConstraint',bestBox, ...
        'PolynomialOrder',bestOrder, 'KernelScale','auto');
else
    SVMtrainedClassifier = fitcsvm(trainX,trainT,'Standardize',true,'KernelFunction',bestKernel,'BoxConstraint',bestBox, ...
        'KernelScale','auto');
end  

% save the trained classifier for further use
save SVMtrainedClassifier;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test the best model on the test dataset - just in development
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Running the best SVM model (test data)");
ypredict = predict(SVMtrainedClassifier,testData); 
TestTarget = table2array(testData(:,1));
%confusionchart(ypredict,TestTarget); % only show when single loop

TN = sum(ypredict+TestTarget == 0);
TP = sum(ypredict+TestTarget == 2);
FN = sum(ypredict+TestTarget == 1 & TestTarget == 1);
FP = sum(ypredict+TestTarget == 1 & TestTarget == 0);

accuracy = (TP + TN) / (TP + TN + FP + FN);
overallAcc = [accuracy TP TN FP FN  ];
disp(overallAcc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test the best model on the train dataset - just in development
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Running the best SVM model (train data)");
ypredict = predict(SVMtrainedClassifier,trainingData); 
TestTarget = table2array(trainingData(:,1));
%confusionchart(ypredict,TestTarget); % only show when single loop

TN = sum(ypredict+TestTarget == 0);
TP = sum(ypredict+TestTarget == 2);
FN = sum(ypredict+TestTarget == 1 & TestTarget == 1);
FP = sum(ypredict+TestTarget == 1 & TestTarget == 0);

accuracy = (TP + TN) / (TP + TN + FP + FN);
overallAcc = [accuracy TP TN FP FN  ];
disp(overallAcc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% grid search function - train a classifier for multiple parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [OverallResults] = RunGridSearch(trainX,trainT,testData)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Overall results will contain:
    % Training Ratio  No of features	Kernel function	box constraint	polynomial order	
    % Train Accuracy	total	TP	TN	FP	FN	
    % Test Accuracy	total	TP	TN	FP	FN	Time Taken
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    OverallResults=[]; % initialise a matrix to hold the grid search results
    
    NoOfTrials=1; % to investigate random variation without a seed
    kernelList = ["polynomial" "linear" "gaussian" "RBF"]; % SVM kernels
    boxList = [ 0.1 0.5 0.7 1 1.2 1.5 2]; % % SVM by default sets the BoxConstraint to be 1,
    orderList = [1 2 3]; 
    %rng(999); 
    disp("Training the SVM model - grid search");
    for trial = 1:NoOfTrials % run multiple trials for same parameters 
        for kernel = kernelList
            for box = boxList
                for order = orderList
                    tic; % start the timer
                    params=[size(trainX,2) kernel box order]; % store parameters for analysis of output
                    if strcmp(kernel,'polynomial') == 1
                        classificationSVM = fitcsvm(...
                            trainX, ...
                            trainT, ...
                            'KernelFunction', kernel, ...
                            'PolynomialOrder', order, ... % order only relevant for polynomial kernel function
                            'KernelScale', 'auto', ...
                            'BoxConstraint', box, ...
                            'Standardize', true, ...
                            'ClassNames', [0; 1]);
                    else
                        classificationSVM = fitcsvm(...
                            trainX, ...
                            trainT, ...
                            'KernelFunction', kernel, ...
                            'KernelScale', 'auto', ...
                            'BoxConstraint', box, ...
                            'Standardize', true, ...
                            'ClassNames', [0; 1]);
                    end                   
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Perform k-fold cross-validation and compute results
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                    partitionedModel = crossval(classificationSVM, 'KFold', 5); % 'stratify' option?
                    % validationPredictions = predicted class output
                    % validationScores = n x 2 (values = confidence that row i is class 0/1)
                    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
                    % Compute validation accuracy
                    %validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError') - matches results below                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % calculate the validation accuracy 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    TN = sum(validationPredictions+trainT == 0);
                    TP = sum(validationPredictions+trainT  == 2);
                    FN = sum(validationPredictions+trainT  == 1 & validationPredictions == 0);
                    FP = sum(validationPredictions+trainT  == 1 & validationPredictions == 1);
                    accuracyTrain = (TP + TN) / (TP + TN + FP + FN);
                    TrainResults = [params accuracyTrain TP + TN + FP + FN TP  TN  FP  FN];
                    dispall=0;
                    if dispall==1
                        disp(TrainResults)
                    end;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % run the test data through the classifier for each case
                    % to check generalisation
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %yfit = SVMtrainedClassifier.predictFcn(testData); %
                    yfit =predict(classificationSVM,testData);
                    % calculate the accuracy of TEST target vs results
                    TestTarget = table2array(testData(:,1));
                    TN = sum(yfit+TestTarget == 0);
                    TP = sum(yfit+TestTarget  == 2);
                    FN = sum(yfit+TestTarget  == 1 & yfit == 0);
                    FP = sum(yfit+TestTarget  == 1 & yfit == 1);
                    accuracyTest = (TP + TN) / (TP + TN + FP + FN);
                    TestResults = [accuracyTest TP + TN + FP + FN TP  TN  FP  FN toc];
                    if dispall==1
                        disp(TestResults );
                    end;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % concatenate results and write to a file
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    OverallResults = [OverallResults; TrainResults TestResults];
                    % confusionchart(yfit,TestTarget); % only show when
                    % required
                    if strcmp(kernel,'polynomial') ~= 1
                        break % only do 1 iteration unless polynomial kernel
                    end
                end
            end
        end
    end
end    


