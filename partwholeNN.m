%Sample script to run a simulation of part or whole reversal of a pair of
%conditional discriminations. Initial training is on discriminations of the
%form AX-1 AY-2 BX-1 BY-2 CX-2 CY-1 DX-2 DY-1 where A and C (and B and D)
%signal the same relationships between X and Y and the two outcomes (1 and 
%2). Following training, each network is cloned and one version is trained
%on a complete reversal where all outcomes are reversed (AX-2 AY-1 BX-2
%BY-1 CX-2 CY-1 DX-1 DY-2) and the other clone is trained on a partial
%reversal where the outcome changes for only one pair of contexts (C/D; 
%AX-1 AY-2 BX-1 BY-2 CX-2 CY-1 DX-1 DY-2).
%This script uses the default network settings defined in createNet and 8
%hidden units. 1000 pairs of cloned network are trained and result averaged
%over them. rmse of the networks' predictions over all training patterns is
%calculated for each epoch of training and plotted.
%Learning rate parameter for the different set of weights are set in this
%script which overrides the default values set in createNet.
%Training data averaged over epochs and over networks are stored in
%dataMean and dataSummary, respectively.
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
tOut = [1 0;... %matrix of output patterns for the initial discimination
        0 1;... %each row is a different pattern
        0 1;... %each column corresponds to an input unit
        1 0;...
        1 0;...
        0 1;...
        0 1;...
        1 0];
wOut = [0 1;... %matrix of output patterns for the whole reversal
        1 0;...
        1 0;...
        0 1;...
        0 1;...
        1 0;...
        1 0;...
        0 1];
pOut = [1 0;... %matrix of output patterns for the partial reversal
        0 1;...
        0 1;...
        1 0;...
        0 1;...
        1 0;...
        1 0;...
        0 1];
trainParam = createSim(tInput, tOut); %wraps the training input and output patterns up in a structure
wholeParam = createSim(tInput, wOut); %wraps the whole reversal input and output patterns up in a structure
partParam = createSim(tInput, pOut); %wraps the partial reversal input and output patterns up in a structure
netParam = createNet(trainParam, nHidden); %creates the network settings
netParam.learnRateIH = .05; %changes the default learning rate parameters
netParam.learnRateOH = .25; %changes the default learning rate parameters
netParam.learnRateHO = .25; %changes the default learning rate parameters
dataStore = zeros(3, runs, trainParam.nEpochs + 1); %matrix to receive training data
%we must create a few separate data matrices to allow the use of parallel
%processing, we will put them back together later
di1 = zeros(runs, 1); %these three arrays are to store the initial rmse of
di2 = zeros(runs, 1); %each network at the begining of a stage of training
di3 = zeros(runs, 1);
ds1 = zeros(runs, trainParam.nEpochs); %these three matrices are to store
ds2 = zeros(runs, trainParam.nEpochs); %rmse data across all epochs of a 
ds3 = zeros(runs, trainParam.nEpochs); %stage of training
parfor i = 1:1:runs %cycle through networks (cloned pairs after training)
    initState = initNet(netParam); %initializes the weight matrices
    di1(i) = mean(sqrt(mean((testNet(netParam, initState, trainParam)' - trainParam.outputPatterns).^2, 2))); %calculate initial rmse
    [netState, ds1(i, :)] = trainNet(netParam, initState, trainParam); %train the network
    di2(i) = mean(sqrt(mean((testNet(netParam, netState, wholeParam)' - wholeParam.outputPatterns).^2, 2))); %calculate initial rmse
    di3(i) = mean(sqrt(mean((testNet(netParam, netState, partParam)' - partParam.outputPatterns).^2, 2))); %calculate initial rmse
    [~, ds2(i, :)] = trainNet(netParam, netState, wholeParam); %train the whole reversal
    [~, ds3(i, :)] = trainNet(netParam, netState, partParam); %train the partial reversal
end
dataStore(1, :, :) = cat(2, di1, ds1); %put together the training data
dataStore(2, :, :) = cat(2, di2, ds2); %put together the whole reversal data
dataStore(3, :, :) = cat(2, di3, ds3); %put together the partial reversal data
clear di1 ds1 di2 ds2 di3 ds3 %clear temporary variables
dataMean = squeeze(mean(dataStore, 2)); %average rmse across networks
dataSummary = squeeze(mean(dataStore, 3)); %average rmse over epochs
plot(dataMean') %plot congruent vs incongruent acquisition
legend('Acquisition', 'Whole', 'Partial')
m = mean(dataSummary, 2);
s = std(dataSummary, 0, 2);
disp(['whole = ' num2str(m(2)) '(' num2str(s(2)) ')']);
disp(['partial = ' num2str(m(3)) '(' num2str(s(3)) ')']);
[~, p, ci, sts] = ttest(dataSummary(2, :), dataSummary(3, :));
disp(['t(' num2str(sts.df) ') = ' num2str(sts.tstat) '; p = ' num2str(p)...
    '; diff = ' num2str(m(2) - m(3)) '; 95%CI[' num2str(ci(1)) ', ' num2str(ci(2)) ']']);
clear nHidden runs tInput tOut wOut pOut trainParam wholeParam partParam ...
    netParam dataStore m s p ci sts
