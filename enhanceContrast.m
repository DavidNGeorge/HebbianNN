function effectiveW = enhanceContrast(W, NNparam)
%% effectiveW = enhanceContrast(W, NNparam)
%
% Enhance network contrast by transforming weights using a sigmoid
% function used by O'Reilly and Manakata (2000):
% eW = 1 / 1 + (offset (Wij / 1 - Wij))^-gain
% Inputs:
%   W = weight matrix
%   NNparam = structure containing network parameters - .enhanceContrast,
%   .weightOffset, and .weightGain
% Output:
%   effectiveW = effective weight matrix after enhancement has been applied
%
% David N. George, Dec 2017
if strcmp(NNparam.enhanceContrast, 'yes')
    effectiveW = 1 ./ (1 + (NNparam.weightOffset .* (W ./ (1 - W))).^(-NNparam.weightGain));
else
    effectiveW = W;
end