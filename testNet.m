function testBlock = testNet(NNParam, NNState, SParam)
%% testBlock = testNet(NNParam, NNState, SParam)
%
% Presents test patterns to a Hebbian network
% Run through all of the training patterns, without the outputs locked on 
% or off so that we can calculate the output unit activation in response to 
% the input patterns alone - ie test the predictions of the network.
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
    NNState.effectiveW_ih = enhanceContrast(NNState.W_ih, NNParam);
    NNState.effectiveW_ho = enhanceContrast(NNState.W_ho, NNParam);
    NNState.currentHidden = activationFunction((NNState.currentInput * ...
        NNState.effectiveW_ih) + (rand(1, NNParam.nHiddenUnits) * ...
        NNParam.noiseLevel), NNParam);
    NNState.currentHidden = wtaFunction(NNState.currentHidden, NNParam);
    NNState.currentOutput = activationFunction((NNState.currentHidden * ...
        NNState.effectiveW_ho) + (rand(1, NNParam.nOutputUnits) * ...
        NNParam.noiseLevel), NNParam);
    NNState.currentOutput = wtaFunction(NNState.currentOutput, NNParam);
    testBlock(:, tt) = NNState.currentOutput;
end