  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Judith Grieves - Neural Computing Coursework - March 2020
% Train an MLP classifier
% Script generated by MATLAB Neural Fitting app
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("### Running MLPtrainClassifier.m");
clear all;
close all; 
display=false; % set to display/suppress diagnostics/results
testing=false; % if true, write out a test results file.
OverallResults=[]; % initialise a results matrix

TestResultsFile="MLPOutputTargetInput.csv" ; % set the results file used
if testing == true
    disp("Test Results File: " + TestResultsFile)
end;

InputFile="Train-breast-cancer-coded.csv"; % set and show the input file used
disp("Input File: " + InputFile);
AllData = readtable(InputFile); 
AllData = table2array(AllData); % convert and lose column headers

rng(999); % set the random seed so results are reproducible

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subsets of data - all features are giving the best overall results for MLP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
features=[2:10]; % = malig/nodes
bc_input = AllData(:,features);  % input data for selected features
bc_target = AllData(:,1);  % target output value 0/1

x = bc_input'; % algorithm requires row-wise features
t = bc_target';

% Choose a Training Function

%trainFcn = 'trainbr';   % (2) 
trainFcn = 'trainlm';   % (1) best result: 86% (fastest) 7s
TrainFcnNum =1; % code the training function to record in output

% set up the test/val/training ratios
trainN=70/100;
valN=15/100;
testN=15/100;

tic % set the clock running to time the trials

rng(1); % set the random seed 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a Fitting Network for a number of params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layerList = [10 15 20]; % run for a selection of layer sizes 
epochList = [ 3 4 5 ]; % no of iterations
muList = [0.1 0.01  0.001 ]; % momentum
%net.trainParam.goal = 0.01; % defaulted to zero
disp('Training the MLP model - grid search');
for hiddenLayerSize = layerList, % new randomisation of train/test split occurs for each new layer size
    for epoch = epochList,  
        for mu = muList, % vary the learning rate           
            net = fitnet(hiddenLayerSize,trainFcn);   % create the network
            % set the parameters            
            net.trainParam.epochs = epoch; %default = 1000
            net.trainParam.mu = mu;  
            net.layers{1}.transferFcn = 'tansig';
            net.layers{2}.transferFcn = 'tansig';
            net.performFcn = 'mse';  % Mean Squared Error performance function
            
            % divide the Data for Training, Validation, Testing
            net.divideFcn = 'dividerand';  % Divide data randomly
            net.divideMode = 'sample';  % Divide up every sample
            net.divideParam.trainRatio = trainN;
            net.divideParam.valRatio = valN;
            net.divideParam.testRatio = testN;

            % store the parameters for checking results
            HyperParams = [ net.divideParam.trainRatio net.divideParam.valRatio net.divideParam.testRatio net.trainParam.epochs net.trainParam.mu net.trainParam.mu_dec net.trainParam.mu_inc net.trainParam.mu_max   net.trainParam.max_fail                 TrainFcnNum hiddenLayerSize];

            [net,tr] = train(net,x,t);  % Train the Network

            % divide the dataset up according to the split ratio
            xTrain = x(tr.trainInd); 
            xVal = x(tr.valInd); 
            xTest = x(tr.testInd); 
            tTrain = t(tr.trainInd); 
            tVal = t(tr.valInd); 
            tTest = t(tr.testInd); 
            
            % Test the Network
            y = net(x); % using whole dataset??!!
            e = gsubtract(t,y);
            
            % classifier results for test
            yTrain  = y(tr.trainInd);
            yVal    = y(tr.valInd);
            yTest   = y(tr.testInd);
            
            % Plots
            if display,
                figure, plotperform(tr)
            end

            % accuracy calculations 
            % bc_target = 0/1 target results
            % y = NN test results 0 <= y <= 1
            TN = sum(round(y')+t' == 0);
            TP = sum(round(y')+t'== 2);
            FN = sum(round(y')+t' == 1 & t' == 1);
            FP = sum(round(y')+t' == 1 & t' == 0);

            accuracy = (TP + TN) / (TP + TN + FP + FN);
            overallAcc = [TP TN FP FN accuracy ];
            
            TN = sum(round(yTest')+tTest' == 0);
            TP = sum(round(yTest')+tTest' == 2);
            FN = sum(round(yTest')+tTest' == 1 & tTest' == 1);
            FP = sum(round(yTest')+tTest' == 1 & tTest' == 0);

            accuracyTest = (TP + TN) / (TP + TN + FP + FN);
            TestAcc = [TP TN FP FN accuracyTest ];
            
            TN = sum(round(yTrain')+tTrain' == 0);
            TP = sum(round(yTrain')+tTrain' == 2);
            FN = sum(round(yTrain')+tTrain' == 1 & tTrain' == 1);
            FP = sum(round(yTrain')+tTrain' == 1 & tTrain' == 0);

            accuracyTrain = (TP + TN) / (TP + TN + FP + FN);
            TrainAcc = [TP TN FP FN accuracyTrain ];
            
            TN = sum(round(yVal')+tVal' == 0);
            TP = sum(round(yVal')+tVal' == 2);
            FN = sum(round(yVal')+tVal' == 1 & tVal' == 1);
            FP = sum(round(yVal')+tVal' == 1 & tVal' == 0);

            accuracyVal = (TP + TN) / (TP + TN + FP + FN);
            ValAcc = [TP TN FP FN accuracyVal ];

            % suppress while doing multi trials
            if display,
                figure, plotconfusion(round(yTrain),tTrain); % plot a confusion matrix - training data
                figure, plotconfusion(round(yVal), tVal); % plot a confusion matrix - test data
                figure, plotconfusion(round(yTest),  tTest); % plot a confusion matrix - validation data                
            end        

            % Record the results of each trial for later analysis
            Results = [ HyperParams overallAcc TestAcc  ValAcc TrainAcc];
            OverallResults = [OverallResults;  Results];
            
            % store the params to analyse results
            HyperParams = [ net.divideParam.trainRatio net.divideParam.valRatio net.divideParam.testRatio net.trainParam.epochs net.trainParam.mu net.trainParam.mu_dec net.trainParam.mu_inc net.trainParam.mu_max   net.trainParam.max_fail                 TrainFcnNum hiddenLayerSize];
        end;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% display training results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputFile= 'MLPParamsAccuracy.csv';
disp("Accuracy Results File: " + outputFile);
disp("training function: " + trainFcn );
disp("activation function 1: " + net.layers{1}.transferFcn)
disp("activation function 2: " + net.layers{2}.transferFcn)
disp("Min overall accuracy: " + min(OverallResults(:,16)));
disp("Max overall accuracy: " + max(OverallResults(:,16)));
disp("Time taken=" + toc );

writematrix(OverallResults,outputFile);  % write out grid search results

if testing == true
    OutputData = [y' bc_target bc_input]; % predicted - target/known - input data
    writematrix(OutputData,TestResultsFile)  % change for different ratios
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  select the hyperparameters of the best predictions and train a final model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accMean = OverallResults(:,16); % overall mean accuracy of test/train
maxVal = max(accMean); % highest overall accuracy
BestResults = OverallResults(accMean== max(accMean),:); % result rows with the highest overall accuracy

BestLayerSize =BestResults(1,11);
BestEpoch =BestResults(1,4);
BestMu =BestResults(1,5);

disp("creating best MLP model: " +  num2str(maxVal) + " : " + BestLayerSize+ " : "  + BestMu + " : " + BestEpoch );

net = fitnet(BestLayerSize,trainFcn);   % create the final network
% set the parameters            
net.trainParam.epochs = BestEpoch; %default = 1000
net.trainParam.mu = BestMu; 
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.performFcn = 'mse';  % Mean Squared Error performance function
rng(999); % set the random seed so results are reproducible
net = train(net,x,t);  % train the final Network

plotperform(tr) % show the performance plot of the chosen classifier
%view(net);  % show a GUI of the network

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test the Network on all the training data and show results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = net(x); 
ypredict = round(y);

TN = sum(ypredict'+t' == 0);
TP = sum(ypredict'+t' == 2);
FN = sum(ypredict'+t' == 1 & t' == 1);
FP = sum(ypredict'+t' == 1 & t' == 0);

TrainAccuracy = (TP + TN) / (TP + TN + FP + FN);
disp("Accuracy on the training data: " + TrainAccuracy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save the trained classifier for future use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MLPtrainedClassifier = net; 
save MLPtrainedClassifier; 
%disp(MLPtrainedClassifier);