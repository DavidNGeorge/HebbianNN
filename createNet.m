function  netParam = createNet(simParam, nHid)
%%netParam = createSim(simParam, nHid)
%
% Creates a three-layer network for Hebbian learning with reciprocal
% connections between the hidden and output layers.
% Inputs:
%   simParam = structure containing simulation parameters -
%   .trainingPatterns and .outputPatterns
%   nHid = number of units in the hidden layer of the network
% Output:
%   netParam = network parameter structure
%
% David N. George, Dec 2017

% Some basic network parameters
netParam.nHiddenUnits = nHid;
netParam.nInputUnits = size(simParam.trainingPatterns, 2);
netParam.nOutputUnits = size(simParam.outputPatterns, 2);
netParam.learnRateIH = 0.1; %learning rate for input-hidden weights
netParam.learnRateOH = 0.1; %learning rate for output-hidden weights
netParam.learnRateHO = 0.1; %learning rate for hidden-output weights
netParam.noiseLevel = 0.05; %noise applied to (non-clamped) unit activations
% Learning rule
% 'cpca': conditional principle components analysis
% 'oja': Oja's rule
netParam.lRule = 'cpca';
% Activation function
% 'linear': out = in
% 'sigmoid': out = 1/1+e^(-gain * (in - offset))
netParam.activationFunction = 'linear';
netParam.sigActFunGain = 2;
netParam.sigActFunOffset = 0;
% Winner-takes-all function
% 'none': no competition
% 'simple': winner preserved, all others = 0
% 'wta': most active = 1; others = 0
% 'power': activation scaled to maximum activation and raised to some power
% set by .wtaParam
% 'powernorm': like 'power', but activity across layer scaled to maintain
% constant total activity = 1
netParam.wtaFunction = 'power';
netParam.wtaParam = 4;
% Renormalization of weights for cpca learning rule
% 'yes': dWij = e[YjXi(m - Wij) + Yj(1 - Xi)(0 - Wij)]
% where m is the renormalization coefficient; renorm1: input-hidden
% weights, renorm2: output_hidden, renorm3 = hidden-output weights.
% 'no': dWij = e[Yj(Xi - Wij)]
% Renormalization depends upon the expected activity in each layer
% determined by formula m = .5/a where a is expected correlation
% Renormalization scaled by sAvgCor param where 0 = no renorm and 1 = max
% renorm. a = .5 - sAvgCor * (.5 - expected correlation)
% Method used by O'Reilly and Manakata (2000)
netParam.renormalize = 'yes';
sAvgCor = 1; %degree of renormalization
netParam.renormIH = .5 / (.5 - (sAvgCor * (.5 - mean(mean(simParam.trainingPatterns))))); %input-hidden based on input sparsity
netParam.renormOH = .5 / (.5 - (sAvgCor * (.5 - mean(mean(simParam.outputPatterns))))); %output-hidden based on output sparsity
netParam.renormHO = .5 / (.5 - (sAvgCor * (.5 - (1 / nHid)))); %hidden-output assumes one winner at hidden layer
% Network contrast enhancement (prior to pattern presentation and applies 
% to weights, not unit activations - i.e. not part of the wta algorithms)
% 'yes': enhance network contrast by transforming weights using a sigmoid
% function: eW = 1 / 1 + (offset (Wij / 1 - Wij))^-gain
% When offset and gain both equal 1, function is linear
% Method used by O'Reilly and Manakata (2000)
netParam.enhanceContrast = 'no';
netParam.weightOffset = 2;
netParam.weightGain = 2;