%Sample script to run a simulation of two concurrent conditional
%discriminations of the form AX-1 AY-2 BX-1 BY-2 CX-2 CY-1 DX-2 DY-1 
%where A and C (and B and D) signal the same relationships between X and Y
%and the two outcomes (1 and 2)
%This script uses the default network settings defined in createNet and the
%minimum number of hidden units required to learn equivalence between pairs
%of stimuli (4; acx-1, acy-2, bdx-2, bcy-1)
%The weight matricies at the end of training are stored in the structure
%netState
%
% David N. George, Dec 2017
nHidden = 4; %number of hidden units
tInput = [1 0 0 0 1 0;... %matrix of input patterns
          1 0 0 0 0 1;... %each row is a different pattern
          0 1 0 0 1 0;... %each column corresponds to an input unit
          0 1 0 0 0 1;...
          0 0 1 0 1 0;...
          0 0 1 0 0 1;...
          0 0 0 1 1 0;...
          0 0 0 1 0 1];
tOut = [1 0;... %matrix of output patterns
        0 1;... %each row is a different pattern
        0 1;... %each column corresponds to an input unit
        1 0;...
        1 0;...
        0 1;...
        0 1;...
        1 0];
trainParam = createSim(tInput, tOut); %wraps the input and output units up in a structure
netParam = createNet(trainParam, nHidden); %creates the network settings
initState = initNet(netParam); %initializes the weight matrices
[netState, ~] = trainNet(netParam, initState, trainParam); %trains the network
%heatmap(netState.W_ih) %draws a nice heatmap of the input-hidden weight matrix - only compatible with R2017a or later
clear i initState netParam nHidden runs teInput teOut testParam tInput tOut trainParam %clean up the workspace