% Judith Grieves - Neural Computing Coursework - March 2020
% Test an SVM classifier

disp("### Running SVMtest.m");
disp("Testing the SVM model");
load SVMtrainedClassifier;
%disp(SVMtrainedClassifier);


Filelist=["Test-breast-cancer-coded.csv","Train-breast-cancer-coded.csv","breast-cancer-coded.csv"];
for InputFile = Filelist, 
    disp("Input File: " + InputFile);
    testData = readtable(InputFile); % whole table as the classifier is choosing the subset of features

    % test the classifier on the test set
    ypredict = predict(SVMtrainedClassifier,testData);
    TestTarget = table2array(testData(:,1));
    figure,confusionchart(TestTarget,ypredict');  

    TN = sum(ypredict+TestTarget == 0);
    TP = sum(ypredict+TestTarget == 2);
    FN = sum(ypredict+TestTarget == 1 & TestTarget == 1);
    FP = sum(ypredict+TestTarget == 1 & TestTarget == 0);

    accuracy = (TP + TN) / (TP + TN + FP + FN);
    overallAcc = [accuracy TP TN FP FN  ];
    disp(overallAcc);
end;
