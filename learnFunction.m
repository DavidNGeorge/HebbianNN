function NNstate = learnFunction(NNparam, NNstate)
%% NNstate = learnFunction(NNparam, NNstate)
%
% Applies the learning rule specified in [NNparam.lRule] to the weight 
% matrix [NNstate.W_ih] for given input vector [NNstate.currentInput] and 
% hidden unit activation [NNstate.currentHidden]
% Conditional PCA [cpca] and Oja's (oja) rules will not differ when basic 
% wta function is used. Does not implement renormalization for Oja's rule.
% Inputs:
%   NNparam = structure containing network parameters
%   NNstate = structure containing weight matrices
% Output:
%   NNstate = structure containing weight matrices
%
% David N. George, Dec 2017
switch NNparam.lRule
    case 'cpca'
        %using slow nested loops to make code easier to read
        if strcmp(NNparam.renormalize, 'yes')
            % With renormalization
            % dWij = e[YjXi(m - Wij) + Yj(1 - Xi)(0 - Wij)]
            % where m is the renormalization coefficient
            for hh = 1:1:NNparam.nHiddenUnits
                for ii = 1:1:NNparam.nInputUnits
                    NNstate.W_ih(ii, hh) = NNstate.W_ih(ii, hh) + NNparam.learnRateIH * ...
                    ((NNstate.currentHidden(hh) * NNstate.currentInput(ii) * ...
                    (NNparam.renormIH - NNstate.W_ih(ii, hh))) + (NNstate.currentHidden(hh) * ...
                    (1 - NNstate.currentInput(ii)) * ( 0 - NNstate.W_ih(ii, hh))));
                end
                for oo = 1:1:NNparam.nOutputUnits
                    NNstate.W_oh(oo, hh) = NNstate.W_oh(oo, hh) + NNparam.learnRateOH * ...
                    ((NNstate.currentHidden(hh) * NNstate.currentOutput(oo) * ...
                    (NNparam.renormOH - NNstate.W_oh(oo, hh))) + (NNstate.currentHidden(hh) * ...
                    (1 - NNstate.currentOutput(oo)) * ( 0 - NNstate.W_oh(oo, hh))));
                end
            end
            for oo = 1:1:NNparam.nOutputUnits
                for hh = 1:1:NNparam.nHiddenUnits
                    NNstate.W_ho(hh, oo) = NNstate.W_ho(hh, oo) + NNparam.learnRateHO * ...
                    ((NNstate.currentOutput(oo) * NNstate.currentHidden(hh) * ...
                    (NNparam.renormHO - NNstate.W_ho(hh, oo))) + (NNstate.currentOutput(oo) * ...
                    (1 - NNstate.currentHidden(hh)) * (0 - NNstate.W_ho(hh, oo))));
                end
            end
        else
            % Without renormalization
            % dWij = e[Yj(Xi - Wij)]
            for hh = 1:1:NNparam.nHiddenUnits
                for ii = 1:1:NNparamInputUnits
                    NNstate.W_ih(ii, hh) = NNstate.W_ih(ii, hh) + NNparam.learnRateIH * ...
                    (NNstate.currentHidden(hh) * (NNstate.currentInput(ii) - NNstate.W_ih(ii, hh)));
                end
                for oo = 1:1:NNparam.nOutputUnits
                    NNstate.W_oh(oo, hh) = NNstate.W_oh(oo, hh) + NNparam.learnRateOH * ...
                    (NNstate.currentHidden(hh) * (NNstate.currentOutput(oo) - NNstate.W_oh(oo, hh)));
                end
            end
            for oo = 1:1:NNparam.nOutputUnits
                for hh = 1:1:NNparam.nHiddenUnits
                    NNstate.W_ho(hh, oo) = NNstate.W_ho(hh, oo) + NNparam.learnRateHO * ...
                    (NNstate.currentOutput(oo) * (NNstate.currentHidden(hh) - NNstate.W_ho(hh, oo)));
                end
            end
        end
    case 'oja'
        % Oja's rule - enhances forgetting, but will have no effect if
        % using ON/OFF wta since in that case we will just be multiplying
        % the weights by 1. In that case, Oja will be just the same as cpca
        % dWij = e[Yj(Xi -WijYj)]
        for hh = 1:1:NNparam.nHiddenUnits
            for ii = 1:1:NNparam.nInputUnits
                NNstate.W_ih(ii, hh) = NNstate.W_ih(ii, hh) + NNparam.learnRateIH * ...
                (NNstate.currentHidden(hh) * (NNstate.currentInput(ii) - ...
                (NNstate.W_ih(ii, hh)) * NNstate.currentHidden(hh)));
            end
            for oo = 1:1:NNparam.nOutputUnits
                NNstate.W_oh(oo, hh) = NNstate.W_oh(oo, hh) + NNparam.learnRateOH * ...
                (NNstate.currentHidden(hh) * (NNstate.currentOutput(oo) - ...
                (NNstate.W_oh(oo, hh)) * NNstate.currentHidden(hh)));
            end
        end
        for oo = 1:1:NNparam.nOutputUnits
            for hh = 1:1:NNparam.nHiddenUnits
                NNstate.W_ho(hh, oo) = NNstate.W_ho(hh, oo) + NNparam.learnRateHO * ...
                (NNstate.currentOutput(oo) * (NNstate.currentHidden(hh) - ...
                (NNstate.W_ho(hh, oo)) * NNstate.currentOutput(oo)));
            end
        end 
end