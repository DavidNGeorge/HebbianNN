function testBlock = primeNet(NNParam, NNState, SParam)
%% testBlock = testNet(NNParam, NNState, SParam)
%
% Presents test patterns to Hebbian network
% Runs through all of the training patterns, applying activation to both 
% the input and output layers, and calculated hidden unit activity. Then, 
% the outputs are turned off and activity is propogated from the hidden 
% units to the output. Useful for testing the effect of 'priming' the
% hidden units by presenting the output briefly.
% Inputs:
%   NNparam = structure containing network parameters
%   NNstate = structure containing weight matrices
%   SParam = structure containing simulation parameters
% Output:
%   testblock = matrix containing output activation for each training
%   pattern
%
% David N. George, Dec 2017

% First allocate space to a matrix to store the predictions for each 
% pattern
testBlock = zeros(NNParam.nOutputUnits, SParam.nTrainingPatterns);
% Run through each pattern in turn - no need to randomize order since
% no learning takes place in the this run
for tt = 1:1:SParam.nTrainingPatterns
    NNState.currentInput = SParam.trainingPatterns(tt, :);
    NNState.currentOutput = SParam.outputPatterns(tt, :);
    NNState.effectiveW_ih = enhanceContrast(NNState.W_ih, NNParam);
    NNState.effectiveW_oh = enhanceContrast(NNState.W_oh, NNParam);
    NNState.effectiveW_ho = enhanceContrast(NNState.W_ho, NNParam);
    NNState.currentHidden = activationFunction((...
            (NNState.currentInput * NNState.effectiveW_ih) + ...
            (NNState.currentOutput * NNState.effectiveW_oh) + ...
            (rand(1, NNParam.nHiddenUnits) * NNParam.noiseLevel)), NNParam);
    NNState.currentHidden = wtaFunction(NNState.currentHidden, NNParam);
    NNState.currentOutput = activationFunction((NNState.currentHidden * ...
        NNState.effectiveW_ho) + (rand(1, NNParam.nOutputUnits) * ...
        NNParam.noiseLevel), NNParam);
    NNState.currentOutput = wtaFunction(NNState.currentOutput, NNParam);
    testBlock(:, tt) = NNState.currentOutput;
end