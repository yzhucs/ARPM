function [shock, v_now] = extract_invariants(dx, garch_1_1_param)
% function [shock, v_now] = extract_invariants(dx, garch_1_1_param)
% parameters:
% - dx: a vector generated by the following GARCH(1,1) process.
%   dx(t) = mu + sigma(t)*epsilon(t)
%   sigma(t)^2 = c + b*sigma(t-1)^2 + a*(dx(t-1)-mu)^2
% - garch_1_1_param: the vector of GARCH(1, 1) parameters.
%   garch_1_1_param(1): the offset.
%   garch_1_1_param(2): the constant variance.
%   garch_1_1_param(3): the GARCH component coefficient.
%   garch_1_1_param(4): the ARCH component coefficient.
%   garch_1_1_param(5): the presample conditional variance, i.e., sigma0^2. 
% returns:
% - shock: the vector of epsilon(t) extracted from the data dx.
% - v_now: the current variance.
%
% Author: Yao Zhu (yzhucs@gmail.com)

% Rename the parameters for readability.
mu = garch_1_1_param(1);
c = garch_1_1_param(2);
b = garch_1_1_param(3);
a = garch_1_1_param(4);
v0 = garch_1_1_param(5);

shock = zeros(size(dx));
for t = 1:length(dx)
  if t == 1
    vt_minus_1 = v0;
    vt = c + b*vt_minus_1; % assuming zero presample shock.
  else
    vt_minus_1 = vt;
    vt = c + b*vt_minus_1 + a*(dx(t-1)-mu)^2;
  end
  shock(t) = (dx(t) - mu) / sqrt(vt);
end

% Remember v_now.
v_now = vt;

return;