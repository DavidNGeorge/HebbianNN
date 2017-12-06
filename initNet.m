function netState = initNet(netParam)
%%function netState = initNet(netParam)
%
% Initializes network for Hebbian learning by generating random weight
% matrices.
% Input:
%   netParam = structure containing network parameters - .nInputUnits,
%   .nHiddenUnits, .nOutputUnits
% Ouput:
%   netState = structure containing weight matrices
%
% David N. George, Dec 2017
netState.W_ih = rand(netParam.nInputUnits, netParam.nHiddenUnits);
netState.W_oh = rand(netParam.nOutputUnits, netParam.nHiddenUnits);
netState.W_ho = rand(netParam.nHiddenUnits, netParam.nOutputUnits);