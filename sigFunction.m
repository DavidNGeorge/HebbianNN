function outA = sigFunction(inA, gain, inflection)
%% outA = sigFunction(inA, gain, inflection)
%
% Sigmoid function with gain and inflection parameters
% Gain is the slope of the curve at the point inA = inflection
% Inputs:
%   inA = array of values of activation of each unit within a layer
%   gain = gain of sigmoid function
%   inflection = inflection point of sigmoid function
% Output:
%   outA = array of values of activation of each unit within a layer
%
% David N. George, Dec 2017
outA = 1 ./ (1 + exp(-gain * (inA - inflection)));