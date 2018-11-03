function garch_1_1_param = fit_garch_1_1(dx)
% function garch_1_1_param = fit_garch_1_1(dx)
% It's a wrapper to call MATLAB GARCH model for estimation.
% parameters:
% - dx: a vector generated by the following GARCH(1,1) process.
%   dx(t) = mu + sigma(t)*epsilon(t)
%   sigma(t)^2 = c + b*sigma(t-1)^2 + a*(dx(t-1)-mu)^2
%   In the above, epsilon(t) is assumed to be standard normal.
% returns:
% - garch_1_1_param(1): the offset.
% - garch_1_1_param(2): the constant variance.
% - garch_1_1_param(3): the GARCH component coefficient.
% - garch_1_1_param(4): the ARCH component coefficient.
% - garch_1_1_param(5): the presample conditional variance, i.e., sigma0^2.
%
% Author: Yao Zhu (yzhucs@gmail.com)

% Initialize a GARCH model with the need to estimate the offset.
Mdl = garch('GARCHLags', 1, 'ARCHLags', 1, 'Offset', NaN);
EstMdl=estimate(Mdl, dx, 'Display', 'off');

% Extract the parameters.
mu = EstMdl.Offset;
c = EstMdl.Constant;
b = EstMdl.GARCH{1};
a = EstMdl.ARCH{1};
% Refit the presample conditional variance as the average sqaure of the offset
% adjusted dx.
v0 = sum((dx - mu).^2) / length(dx);

garch_1_1_param = [mu, c, b, a, v0];

return;