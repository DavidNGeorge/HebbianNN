function [NNState, LData] = trainNet(NNParam, NNState, SParam)
%% [NNState, LData] = trainingFunction(NNParam, NNState, SParam)
%
% Trains a Hebbian network.
% Inputs:
%   NNparam = structure containing network parameters
%   NNstate = structure containing weight matrices
%   SParam = structure containing simulation parameters
% Output:
%   NNstate = structure containing weight matrices
%   LData = root mean square error over all training patterns for each
%   epoch of training
%
% David N. George, Dec 2017

%pre-allocate space to array for training data (rmse)
LData = zeros(1, SParam.nEpochs);
%cycle through epochs
for ee = 1:1:SParam.nEpochs
    %generate trial sequence depending on randomOrder flag
    if strcmp(SParam.randomOrder, 'yes')
            trialOrder = randperm(SParam.nTrainingPatterns);
    else
            trialOrder = 1:1:SParam.nTrainingPatterns;
    end
    %run through a set of training patterns
    for tt = 1:1:SParam.nTrainingPatterns
        %each training trial comprises six steps:
        %1. Select input pattern
        %2. Select output pattern
        %3. Apply contrast enhancement to weight matrices 
        %4. Propogate activity to hidden units and apply noise if required
        %5. Apply winner-takes-all algorithm
        %6. Apply learning rule
        %NB activity not propogated to output units because they are
        %locked on or off on training trial by output feedback
        NNState.currentInput = SParam.trainingPatterns(trialOrder(tt), :);
        NNState.currentOutput = SParam.outputPatterns(trialOrder(tt), :);
        NNState.effectiveW_ih = enhanceContrast(NNState.W_ih, NNParam);
        NNState.effectiveW_oh = enhanceContrast(NNState.W_oh, NNParam);
        NNState.currentHidden = activationFunction((...
            (NNState.currentInput * NNState.effectiveW_ih) + ...
            (NNState.currentOutput * NNState.effectiveW_oh) + ...
            (rand(1, NNParam.nHiddenUnits) * NNParam.noiseLevel)), NNParam);
        NNState.currentHidden = wtaFunction(NNState.currentHidden, NNParam);
        NNState = learnFunction(NNParam, NNState);
    end
    testBlock = testNet(NNParam, NNState, SParam);
    %here mean rmse is calculated - that is, rmse is calculated over the
    %output untils for each pattern separately, then the average of all
    %those is calculated for the epoch
    LData(ee) = mean(sqrt(mean((testBlock' - SParam.outputPatterns).^2, 2)));
end