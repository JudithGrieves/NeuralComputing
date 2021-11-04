% Judith Grieves - Neural Computing Coursework - March 2020
% Test an MLP classifier

disp("### Running MLPtest.m");
disp("Testing the MLP model");
load MLPtrainedClassifier;
%disp(MLPtrainedClassifier);

Filelist=["Test-breast-cancer-coded.csv","Train-breast-cancer-coded.csv","breast-cancer-coded.csv"];
for InputFile = Filelist, 
    disp("Input File: " + InputFile);
    testData = readtable(InputFile);% read in test data
    testData = table2array(testData);% Convert to array
    features=[2:10]; % vary features - all is [2:10] [3 5 6 9]
    x = testData(:,features)';

    % test on the dataset
    y = MLPtrainedClassifier(x);
    ypredict = round(y);
    TestTarget = testData(:,1);
    figure,confusionchart(TestTarget,ypredict'); 

    TN = sum(ypredict'+TestTarget == 0);
    TP = sum(ypredict'+TestTarget == 2);
    FN = sum(ypredict'+TestTarget == 1 & TestTarget == 1);
    FP = sum(ypredict'+TestTarget == 1 & TestTarget == 0);

    accuracy = (TP + TN) / (TP + TN + FP + FN);
    overallAcc = [accuracy TP TN FP FN  ];
    disp(overallAcc);
end;
