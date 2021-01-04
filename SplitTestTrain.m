%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to split an input file randomly between test and train 
% for a given training ratio
% This ensures that both models use the same data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("### Running SplitTestTrain.m");

trainRatio=80;
testRatio= 100 - trainRatio ;

InputFileName="breast-cancer-coded.csv"; % set and show the input file used
disp("Input File: " + InputFileName); % display the input file
AllData = readtable(InputFileName);  % read in the file
%AllData = table2array(AllData);
class= table2array(AllData(:,1));
disp("Recurrence rate (Overall) : " + round((size(class(class==1),1) / size(class,1)) * 100) ...
    + "%  ")

% randomly split into test & training sets

rng(0); % seed the randomiser for repeatable results
[trainInd,testInd] = dividerand(size(AllData,1),trainRatio,testRatio);
train = AllData(trainInd,:); % create the random train set
test = AllData(testInd,:); % create the random test set
disp("Train/Test Ratio: " + trainRatio + "/" +testRatio);
disp("File count (Train): " + size(train,1) + " (Test): " + size(test,1));

% show the train/test ratio
class= table2array(train(:,1));
disp("Recurrence rate (train) : " + round(size(class(class==1),1) / size(class,1) * 100)+ "%  " )
class= table2array(test(:,1));
disp("Recurrence rate (test): " + round(size(class(class==1),1) / size(class,1) * 100)+ "%  " )

outputFile= 'Train-breast-cancer-coded.csv'; % write the training data to file
disp("Output Train File: " + outputFile); % display the input file
writetable(train,outputFile); 

outputFile= 'Test-breast-cancer-coded.csv'; % write the test data to file
disp("Output Test File: " + outputFile); % display the input file
writetable(test,outputFile); 