function [r, vol, W, meanR, covR] = mv_frontier(R, p, num_alloc)
% function [r, vol, W, meanR, covR] = mv_frontier(R, p, num_alloc)
% This function generates the mean-variance frontier from the return scenarios.
% We constrain no-short in the allocations.
% parameters:
% - R: the JxN matrix of stock ex-ante returns, each row is a scenario.
% - p: the vector of length J, where p(j) is the probability of the j-th scenario.
% - num_alloc: the number of intermediate allocations in between the min-variance
%   and the max-return allocations.
% returns:
% - r: the vector of returns of the allocations on the mean-variance frontier.
% - vol: the vector of volatilities of the allocations on the mean-variance frontier.
% - W: the Nx(num_alloc+2) weight matrix, where each column defines an allocation
%   on the mean-variance frontier.
% - meanR: the mean vector of stock returns under the probability p.
% - covR: the covariance matrix of stock returns under the probability p.
% Reference:
% - This function is adapted from the code EfficientFrontier.m by Attilio Meucci.

warning off;

% Enforce p to be a row vector.
if size(p, 1) > size(p, 2)
  p = p';
end

[J, N] = size(R);

% Compute the mean and covariance of stock returns.
meanR = (p * R)'; % make it a column vector.
Scnd_Mom = (R' .* (ones(N,1)*p)) * R; Scnd_Mom = (Scnd_Mom + Scnd_Mom') / 2;
covR = Scnd_Mom - meanR * meanR';

% Setup the constraints on the allocation weights.
Aeq = ones(1,N);
beq = 1;
lb = zeros(N, 1); % lower bound for no short.
% Empty placeholders.
A = []; b = []; ub = []; x0=[];
% No display.
options = optimset('Display', 'off');

% Determine the min-variance allocation.
H_min_var = covR;
f_min_var = zeros(N,1);
w_min_var = quadprog(H_min_var, f_min_var, A, b, Aeq, beq, lb, ub, x0, options);
r_min_var = w_min_var' * meanR;
vol_min_var = sqrt(w_min_var' * covR * w_min_var);

% Determine the max-return allocation.
f_max_r = -meanR;
w_max_r = linprog(f_max_r, A, b, Aeq, beq, lb, ub, options);
r_max_r = w_max_r' * meanR;
vol_max_r = sqrt(w_max_r' * covR * w_max_r);

% Determine the intermediate allocations.
r = zeros(num_alloc + 2, 1);
r(1) = r_min_var;
r(end) = r_max_r;
vol = zeros(num_alloc + 2, 1);
vol(1) = vol_min_var;
vol(end) = vol_max_r;
W = zeros(N, num_alloc + 2);
W(:, 1) = w_min_var;
W(:, end) = w_max_r;
% The objective function is fixed to be minimizing the variance.
H = covR;
f = zeros(N,1);
r_step_size = (r_max_r - r_min_var) / (num_alloc + 1);
for i = 2:(num_alloc+1)
  target_r = r_min_var + (i - 1) * r_step_size;
  % Add the targeted return constraint for this step.
  Aeq_step = [Aeq;
              meanR'];
  beq_step = [beq;
              target_r];
  W(:, i) = quadprog(H, f, A, b, Aeq_step, beq_step, lb, ub, x0, options);
  % Calculate r(i) even though it should be satisfied because of the target constraint.
  r(i) = W(:, i)' * meanR;
  vol(i) = sqrt(W(:, i)' * covR * W(:, i));
end

return;