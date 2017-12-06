function outA = wtaFunction(inA, NNparam)
%% outA = wtaFunction(inA)
%
% Winner-takes-all function for Hebbian network
% Inputs:
%   inA = array of values of the activation of each unit within a layer
%   NNparam = structure containing network parameters - .wtaFunction and
%   .wtaParam control aspect of wta function
% Output:
%   outA = array of values of activation level of each unit within the
%   layer once the wta function has been applied
%
% NNparam is created by the function createNet. Currently, only one
% parameter (.wtaParam) is defined in createNet. If new wta functions are
% added to wtaFunction which require more than one parameter, createNet
% must be updated.
%
% David N. George, Dec 2017
switch NNparam.wtaFunction
    case 'none'
        % No wta
        outA = inA;
    case 'simple'
        % Simplest form possible - leave the most active alone, other units 
        % set to 0
        [maxActivation, maxIndex] = max(inA);
        outA = zeros(1, size(inA, 2));
        outA(maxIndex) = maxActivation;
    case 'onoff'
        % Simple binary function - most active unit set to 1, other units 
        % set to 0
        [~, maxIndex] = max(inA);
        outA = zeros(1, size(inA, 2));
        outA(maxIndex) = 1;
    case 'power'
        % This not really a wta function as such, but does introduce some
        % competition and thus enhance the contrast of the activation level 
        % if units by scaling activation by that of the most active unit 
        % and then raising all activations to some powers
        % Similar to formulation used by Wallis & Rolls (1997), but sets
        % max activation to 1 rather than total activity within the layer
        [maxActivation, ~] = max(inA);
        outA = inA ./ maxActivation;
        outA = outA.^NNparam.wtaParam;
    case 'powernorm'
        % Basically the same as power, but activity across hidden layer 
        % normalized to ensure constant level of activity = 1
        % Formulation used by Wallis & Rolls (1997)
        outA = inA.^NNparam.wtaParam / sum(inA.^NNparam.wtaParam);
end