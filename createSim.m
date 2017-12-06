function simParam = createSim(input, output)
%%simParam = createSim(input, output)
%
% Generate simulation parameter structure for use with Hebbian learning 
% network based on some common settings and the supplied input and output 
% matricies
% Inputs:
%   input = matrix of input patterns. Each row is a different pattern, each
%   column is a different input unit.
%   output = matrix of output patterns. Each row is a different pattern,
%   each column is a different input unit.
% Output:
%   simParam = simulation parameter structure
%
% David N. George, Dec 2017

%performs a quick check that there are the same number of rows in the two
%matrices
if size(input, 1) ~= size(output, 1)
    return
end
%wrap the input and output patterns up in the data structure
simParam.trainingPatterns = input;
simParam.outputPatterns = output;
simParam.nTrainingPatterns = size(simParam.trainingPatterns, 1);
%default simulation paramters to randomize the order of patterns within
%each epoch and to train for 50 epochs.
simParam.randomOrder = 'yes';
simParam.nEpochs = 50;