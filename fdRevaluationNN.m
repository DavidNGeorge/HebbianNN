%Sample script to run a simulation of two concurrent conditional
%discriminations of the form AX-1 AY-2 BX-1 BY-2 CX-2 CY-1 DX-2 DY-1 
%where A and C (and B and D) signal the same relationships between X and Y
%and the two outcomes (1 and 2). Following training, the effect of
%revaluing two of the contexts on responding to the other two contexts is
%assessed: A-1; B-2; C-?; D-?.
%This script uses the default network settings defined in createNet and the
%8 hidden units. 1000 networks are trained and result averaged over
%them. Mean responses to each of the four contexts before and after
%revaluation is stored in trainData and testData, respectively. ABData and
%CDData contain resuts from the final test for all networks in an
%ANOVA-friendly format (four coloumns: A/C-O1, A/C-O2, B/D-01, B/D-02).
%Learning rate parameter for the different set of weights are set in this
%script which overrides the default values set in createNet.
%
% David N. George, Jan 2018
nHidden = 8; %number of hidden units
runs = 1000; %number of networks trained
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
reInput = [1 0 0 0 0 0;... %input patterns for revaulation training
           0 1 0 0 0 0];
reOut = [1 0;... %output patterns for revaluation training
        0 1];
teInput = [1 0 0 0 0 0;... %input patterns for testing
           0 1 0 0 0 0;...
           0 0 1 0 0 0;...
           0 0 0 1 0 0];
teOut = [0 0;... %dummy array of output patterns for testing
         0 0;...
         0 0;...
         0 0];
trainParam = createSim(tInput, tOut); %wraps the training input and output patterns up in a structure
revalParam = createSim(reInput, reOut); %wraps the revaluation input and output patterns up in a structure
testParam = createSim(teInput, teOut); %wraps the testing input and output patterns up in a structure
netParam = createNet(trainParam, nHidden); %creates the network settings
revalParam.nEpochs = 2; %changes the default number of epochs of revaluaton training
netParam.learnRateIH = .1; %changes the default learning rate parameters
netParam.learnRateOH = .2; %changes the default learning rate parameters
netParam.learnRateHO = .2; %changes the default learning rate parameters
trainStore  = zeros(netParam.nOutputUnits, testParam.nTrainingPatterns, runs); %create array to store post-training test data
testStore  = zeros(netParam.nOutputUnits, testParam.nTrainingPatterns, runs); %create array to store post-revaluation test data
parfor i = 1:1:runs %cycle through networks
    initState = initNet(netParam); %initializes the weight matrices
    [netState, ~] = trainNet(netParam, initState, trainParam); %train the network
    trainStore(:, :, i) = testNet(netParam, netState, testParam); %test the network
    [reState, ~] = trainNet(netParam, netState, revalParam); %revalue the network
    testStore(:, :, i) = testNet(netParam, reState, testParam); %test the network
end
trainData = squeeze(mean(trainStore, 3))'; %post-training test data averaged over networks
testData = squeeze(mean(testStore, 3))'; %post-revaluation test data averaged over networks
%display some summary data
contextNames = {'A', 'B', 'C', 'D'};
outputNames = {'O1', 'O2'};
T = array2table(trainData, 'RowNames', contextNames, 'VariableNames', outputNames);
disp('pre-revaluation test:')
disp(T)
T = array2table(testData, 'RowNames', contextNames, 'VariableNames', outputNames);
disp('post-revaluation test:')
disp(T)
ABData = squeeze([testStore(1, 1, :) testStore(2, 1, :) testStore(1, 2, :) testStore(2, 2, :)]); %AB test data in an ANOVA-friendly format
CDData = squeeze([testStore(1, 3, :) testStore(2, 3, :) testStore(1, 4, :) testStore(2, 4, :)]); %CD test data in an ANOVA-friendly format
%clear some variables
clear nHidden runs tInput tOut reInput reOut teInput teOut trainParam ...
    revalParam testParam netParam trainStore testStore contextNames ...
    outputNames T