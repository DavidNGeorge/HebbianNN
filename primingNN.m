%priming with outcome before presenting congruent or incongruent context
%pairs.%Sample script to run a simulation of two concurrent conditional
%discriminations of the form AX-1 AY-2 BX-1 BY-2 CX-2 CY-1 DX-2 DY-1 
%where A and C (and B and D) signal the same relationships between X and Y
%and the two outcomes (1 and 2). Following training, pairs of equivalent
%(AC/BD) or distinctive (AD/BC) contexts are presented at test. One or
%other output unit is primed prior to presentation of each context (in
%practice, this is achieved by partially activating an output unit at the
%same time that the input is presented to represent some trace activity).
%This script uses the default network settings defined in createNet and the
%8 hidden units. 1000 networks are trained and result averaged over
%them. Mean responses and absolute deviation scores in response to each 
%combination of context are calculated and stored in testDiff and absDiff.
%
% David N. George, Jan 2018
nHidden = 8; %number of hidden units
runs = 1000; %number of networks trained
prime = .2; %strength of activity for priming
tInput = [1 0 0 0 1 0;... %matrix of input patterns
          1 0 0 0 0 1;... %each row is a different pattern
          0 1 0 0 1 0;... %each column corresponds to an input unit
          0 1 0 0 0 1;...
          0 0 1 0 1 0;...
          0 0 1 0 0 1;...
          0 0 0 1 1 0;...
          0 0 0 1 0 1];
tOut = [1 0;... %matrix of output patterns for the congruent discimination
        0 1;... %each row is a different pattern
        0 1;... %each column corresponds to an input unit
        1 0;...
        1 0;...
        0 1;...
        0 1;...
        1 0];
teInput = [1 0 1 0 0 0;... %input patterns for testing
           0 1 0 1 0 0;...
           1 0 0 1 0 0;...
           0 1 1 0 0 0];
teOut = [0 0;... %dummy array of output patterns for testing
         0 0;...
         0 0;...
         0 0];
prOut1 = [1 0;... %output patterns for priming of O1
          1 0;...
          1 0;...
          1 0] * prime;
prOut2 = [0 1;... %output patterns for priming of O2
          0 1;...
          0 1;...
          0 1] * prime;
trainParam = createSim(tInput, tOut); %wraps the training input and output patterns up in a structure
testParam = createSim(teInput, teOut); %wraps the testing input and output patterns up in a structure
primeO1Param = createSim(teInput, prOut1); %wraps the O1 priming input and output patterns up in a structure
primeO2Param = createSim(teInput, prOut2); %wraps the O2 priming input and output patterns up in a structure
netParam = createNet(trainParam, nHidden); %creates the network settings
netParam.learnRateIH = .1; %changes the default learning rate parameters
netParam.learnRateOH = .2; %changes the default learning rate parameters
netParam.learnRateHO = .2; %changes the default learning rate parameters.nOutputUnits, testParam.nTrainingPatterns, runs);
testStore  = zeros(netParam.nOutputUnits, testParam.nTrainingPatterns, runs); %create array to store test data
parfor i = 1:1:runs %cycle through networks
    initState = initNet(netParam); %initializes the weight matrices
    [netState, ~] = trainNet(netParam, initState, trainParam); %train the network
    testStore(:, :, i) = testNet(netParam, netState, testParam); %test the network
    primeStore1(:, :, i) = primeNet(netParam, netState, primeO1Param); %test with O1 primed
    primeStore2(:, :, i) = primeNet(netParam, netState, primeO2Param); %test with O2 primed
end
%display some summary data - first for the standard test trials with no priming
testNames = {'AC', 'BD', 'AD', 'BC'};
outputNames = {'O1', 'O2'};
dataMean = squeeze(mean(testStore, 3))'; %test data averaged over networks
T = array2table(dataMean, 'RowNames', testNames, 'VariableNames', outputNames);
disp('average activity');
disp(T);
dataVar = squeeze(var(testStore, 0, 3))'; %test data variance over networks
T = array2table(dataVar, 'RowNames', testNames, 'VariableNames', outputNames);
disp('activity variance');
disp(T);
testDiff = squeeze(testStore(1, :, :) - testStore(2, : , :)); %difference in O1 and O2 activity
[netOut, meanDiff, p, ci, sts] = analyseData(testDiff);
disp(['mean difference between O1 and O2 activity: AC/BD =  ' num2str(netOut(1)) '; AD/BC = ' num2str(netOut(2))])
disp(['t(' num2str(sts.df) ') = ' num2str(sts.tstat) '; p = ' num2str(p)...
    '; diff = ' num2str(meanDiff) '; 95%CI[' num2str(ci(1)) ', ' num2str(ci(2)) ']']);
absDiff = squeeze(abs(testStore(1, :, :) - testStore(2, : , :))); %absolute difference in O1 and O2 activity
[netOut, meanDiff, p, ci, sts] = analyseData(absDiff);
disp(['absolute difference between O1 and O2 activity: AC/BD =  ' num2str(netOut(1)) '; AD/BC = ' num2str(netOut(2))])
disp(['t(' num2str(sts.df) ') = ' num2str(sts.tstat) '; p = ' num2str(p)...
    '; diff = ' num2str(meanDiff) '; 95%CI[' num2str(ci(1)) ', ' num2str(ci(2)) ']']);
%repeat the lot with O1 or O2 primed
%first, collapse across the two types of prime so that the data array has
%two columns corresponding to primed and unprimed outcomes
outputNames = {'primed', 'unprimed'};
primeStore = [mean([primeStore1(1, :, :); primeStore2(2, :, :)], 1); mean([primeStore1(2, :, :); primeStore2(1, :, :)], 1)];
dataMean = squeeze(mean(primeStore, 3))'; %test data averaged over networks
T = array2table(dataMean, 'RowNames', testNames, 'VariableNames', outputNames);
disp('priming test:')
disp('average activity');
disp(T);
dataVar = squeeze(var(primeStore, 0, 3))'; %test data variance over networks
T = array2table(dataVar, 'RowNames', testNames, 'VariableNames', outputNames);
disp('activity variance');
disp(T);
primeDiff = squeeze(primeStore(1, :, :) - primeStore(2, : , :)); %difference in P and UP activity
[netOut, meanDiff, p, ci, sts] = analyseData(primeDiff);
disp(['mean difference between primed and unprimed activity: AC/BD =  ' num2str(netOut(1)) '; AD/BC = ' num2str(netOut(2))])
disp(['t(' num2str(sts.df) ') = ' num2str(sts.tstat) '; p = ' num2str(p)...
    '; diff = ' num2str(meanDiff) '; 95%CI[' num2str(ci(1)) ', ' num2str(ci(2)) ']']);
absPrimeDiff = squeeze(abs(testStore(1, :, :) - testStore(2, : , :))); %absolute difference in O1 and O2 activity
[netOut, meanDiff, p, ci, sts] = analyseData(absPrimeDiff);
disp(['absolute difference between primed and unprimed activity: AC/BD =  ' num2str(netOut(1)) '; AD/BC = ' num2str(netOut(2))])
disp(['t(' num2str(sts.df) ') = ' num2str(sts.tstat) '; p = ' num2str(p)...
    '; diff = ' num2str(meanDiff) '; 95%CI[' num2str(ci(1)) ', ' num2str(ci(2)) ']']);
clear nHidden runs tInput tOut reInput reOut teInput teOut trainParam ...
    testParam netParam testStore dataMean dataVar netOut ...
    meanDiff p ci sts testNames outputNames T prime primeO1Param ...
    primeO2Param primeStore primeStore1 primeStore2 prOut1 prOut2

function [netOut, meanDiff, p, ci, sts] = analyseData(dataArray)
    testMean = [mean(dataArray(1:2, :)); mean(dataArray(3:4, :))]; %averaged over equivalent (AC/BD) or distinctive (AD/BC) compounds
    netOut = squeeze(mean(testMean, 2)); %averaged over networks
    meanDiff = netOut(1) - netOut(2); %difference in net output activity between equivalent and distinctive compounds
    [~, p, ci, sts] = ttest(testMean(1, :), testMean(2, :)); %t-test
end