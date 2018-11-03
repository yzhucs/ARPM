function x_next_vec = next_step_rd(garch_1_1_param, v_now, dx_now, x_now, shock_next_vec)
% function x_next_vec = next_step_rd(garch_1_1_param, v_now, dx_now, shock_next_vec)
% This is the function to project the next step risk driver.
% parameters:
% - garch_1_1_param: the vector of GARCH(1, 1) parameters.
%   garch_1_1_param(1): the offset.
%   garch_1_1_param(2): the constant variance.
%   garch_1_1_param(3): the GARCH component coefficient.
%   garch_1_1_param(4): the ARCH component coefficient.
% - v_now: the current variance extracted from data.
% - dx_now: the current delta x, a scalar.
% - x_now: the current risk driver, a scalar.
% - shock_next_vec: a vector of sampled next step shocks.
% returns:
% - x_next_vec: a vector of next step risk drivers.
%
% Author: Yao Zhu (yzhucs@gmail.com)

% Rename the parameters for readability.
mu = garch_1_1_param(1);
c = garch_1_1_param(2);
b = garch_1_1_param(3);
a = garch_1_1_param(4);

% The next step variance.
v_next = c + b*v_now + a*(dx_now - mu)^2;

% The next step risk drivers.
x_next_vec = x_now + mu + sqrt(v_next)*shock_next_vec;

return;