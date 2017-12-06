function outA = activationFunction(inA, NNparam)
%% outA = activationFunction(inA, NNparam)
%
% Activation function for unit in Hebbian network
% Inputs:
%   inA = array of values of net input to each unit within a layer
%   NNparam = structure containing network parameters -
%   .activationFunction, .sigActFunGain and .sigActFunOffset
% Output:
%   outA = array of values of activation level of each unit within the
%   layer
%
% NNparam is created by the function createNet. If any new activation
% functions require additional parameters, createNet will need to be
% modified.
%
% David N. George, Dec 2017
switch NNparam.activationFunction
    case 'linear'
        outA = inA;
    case 'sigmoid'
        outA = sigFunction(inA, NNparam.sigActFunGain, NNparam.sigActFunOffset);
end