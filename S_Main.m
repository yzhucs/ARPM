% Author: Yao Zhu (yzhucs@gmail.com)
%
% README: 1. The code in the following assumes the available of MATLAB garch model fiting, and copula fit.
%         2. The run on the entire dataset of stock prices from DatabaseStocks.mat could be time consuming.
%            For a quick trial run, please set "use_data_small" to 1 in the following script for using only
%            a small subset.
%         3. The "save" statements in the following script make it clear the intermediate results I have saved as
%            .mat files. You can choose to run just a subsection of the code instead of from the scratch, assuming
%            you have loaded the saved intermediate results, and set the necessary environment variables,
%            like the stock price matrix "S", manually.


clc; clear;

% Pin down the RNG from the very beginning for reproducibility.
rng(0);

% set this flag to 1 if only want to use a small dataset with 10 stocks.
use_data_small = 1;

%%%%%%%%%%%%%%%%%Load the data of stock prices.%%%%%%%%%%%%%%%%%
% DatabaseStocks.mat is acquired from AMeucciRiskandAssetAllocationRoutines/Ch3_ModellingMarket/C_DimensionReduction
% Three variables will be loaded:
% - P, the matrix of stock prices, with each column corresponding to the daily
%   prices of a stock.
% - D, the integer number of dates.
% - M, the market index.
load DatabaseStocks.mat;
if use_data_small == 1
  % We will use S as stock price matrix in the following.
  S = P(:, 1:10);
else
  S = P;
end
% Clear off the variables we don't need anymore.
clear P; clear D; clear M;

%%%%%%%%%%%%%%%%%Model the market to extract invariants.%%%%%%%%%%%%%%%%%
% Get the daily changes of the risk drivers x=log(p):
dX = log(S(2:end,:)) - log(S(1:end-1,:));
garch_1_1_params = zeros(size(dX, 2), 5);
Shocks_empirical = zeros(size(dX));
garch_v_now = zeros(size(dX, 2), 1);
for d = 1:size(dX, 2)
  % Fit the GARCH(1,1) parameters for each stock.
  % - garch_1_1_params(d, 1): the offset.
  % - garch_1_1_params(d, 2): the constant variance.
  % - garch_1_1_params(d, 3): the GARCH component coefficient.
  % - garch_1_1_params(d, 4): the ARCH component coefficient.
  % - garch_1_1_params(d, 5): the presample conditional variance, i.e., sigma(0)^2.
  garch_1_1_params(d, :) = fit_garch_1_1(dX(:, d));
  % Extract the invariants.
  [Shocks_empirical(:, d), garch_v_now(d)] = extract_invariants(dX(:, d), garch_1_1_params(d, :));
end
% Save the empirical shocks and current variances.
if use_data_small == 1
  save('empirical_small.mat', 'S', 'garch_1_1_params', 'Shocks_empirical', 'garch_v_now');
else
  save('empirical.mat', 'S', 'garch_1_1_params', 'Shocks_empirical', 'garch_v_now');
end

%%%%%%%%%%%%%%%%%Generate projected scenarios by Copula-Marginal.%%%%%%%%%%%%%%%%%
% Model the marginal distribution of shocks as empirical historical distribution.
U_shocks_empirical = zeros(size(Shocks_empirical)); % The grades of shocks empirical data.
for d = 1:size(Shocks_empirical, 2)
  U_shocks_empirical(:, d) = ecdf_grade(Shocks_empirical(:, d), Shocks_empirical(:, d));
end
% Fit a t-copula to U_shocks_empirical. Because copulafit needs data to be
% strictly within (0, 1), I subtract a small numer 1e-5 from U_shocks_empirical.
% WARNING: This step is really time and memory consuming unless use data small.
% - Rho is the correlation matrix.
% - nu is the degree of freedom.
[Rho_copula, nu_copula] = copulafit('t', U_shocks_empirical - 1e-5, ...
                                    'Method', 'ApproximateML');
%////////////////////////////////////NOTE///////////////////////////////////////
% Because the above processing steps on the entire dataset could be time
% consuming, I have saved the parameters from copulafit on the entire dataset
% in t_copula_param.mat. You can just load Rho_copula, nu_copula from it and
% continue with the following steps. Of course you also need to load the
% empirical.mat, and set use_data_small=0 for the entire dataset from here.
%///////////////////////////////////////////////////////////////////////////////

% Generate the scenarios.
J = 10000;
U_shocks_scenario = copularnd('t', Rho_copula, nu_copula, J); % scenario of grades.
Shocks_scenario = zeros(size(U_shocks_scenario)); % scenario of next step shocks.
X_scenario = zeros(size(U_shocks_scenario)); % scenario of next step risk drivers.
S_scenario = zeros(size(U_shocks_scenario)); % scenario of next step stock prices.
PnL_scenario = zeros(size(U_shocks_scenario)); % scenario of next step P&L.
for d = 1:size(U_shocks_scenario, 2)
  % Invert to find the shocks' scenarios.
  Shocks_scenario(:, d) = inv_ecdf_grade(Shocks_empirical(:, d), ...
                                         U_shocks_scenario(:, d));
  % Get the next day risk driver.                                       
  X_scenario(:, d) = next_step_rd(garch_1_1_params(1:4), garch_v_now(d), ...
                                  dX(end, d), log(S(end,d)), ...
                                  Shocks_scenario(:, d));
  % Get the next day stock prices.
  S_scenario(:, d) = exp(X_scenario(:, d));
  % Get the next day profit and loss.
  PnL_scenario(:, d) = S_scenario(:, d) - S(end,d);
end
% Save the scenarios.
if use_data_small == 1
  save('scenario_small.mat', 'Rho_copula', 'nu_copula', 'U_shocks_scenario', ...
       'Shocks_scenario', 'X_scenario', 'S_scenario', 'PnL_scenario');
else
  save('scenario.mat', 'Rho_copula', 'nu_copula', 'U_shocks_scenario', ...
       'Shocks_scenario', 'X_scenario', 'S_scenario', 'PnL_scenario');
end

%%%%%%%%%%%%%%%%%View processing of the shocks.%%%%%%%%%%%%%%%%%
% We want the posterior p_ to satisfy the following views:
% (1) mean_{p_}(Shocks_scenario(:,1) - Shocks_scenario(:,2)) = -0.03
% (2) mean_{p_}(Shocks_scenario(:,3) - Shocks_scenario(:,2)) = 0.15

% The uniform prior on the scenarios.
J = size(Shocks_scenario, 1);
p0 = ones(J, 1) / J;
% Linear equality constraints encoding the views and normalization condition.
% Aeq*p_ = beq.
Aeq = [(Shocks_scenario(:,1) - Shocks_scenario(:,2))'; ...
       (Shocks_scenario(:,3) - Shocks_scenario(:,2))'; ...
        ones(1, J)];
beq = [-0.03; 0.15; 1];
% Solve for p_ by minimizing the relative entropy subject to the constraints.
% Call the code EntropyProg.m by Attilio.
p_ = EntropyProg(p0, [], [], Aeq, beq);
% Renormalization once more.
p_ = p_ / sum(p_);
% Save the posterior p_ to the scenario file.
if use_data_small == 1
  save('scenario_small.mat', 'p_', '-append');
else
  save('scenario.mat', 'p_', '-append');
end
% Plot the posterior distribution p_ to see its difference from the uniform prior
% because of the views.
figure;
bar([1:length(p_)], p_);
grid on;
set(gca, 'xlim', [1 length(p_)]);
xlabel('scenario index', 'FontSize', 20);
ylabel('probability mass', 'FontSize', 20);
set(gca, 'fontsize', 20);
saveas(gcf, 'p_.jpg');

%%%%%%%%%%%%%%%%%Portfolio Optimization.%%%%%%%%%%%%%%%%%
% We compute the returns under each scenario for each stock.
R_scenario = zeros(size(PnL_scenario)); % scenario of returns.
for d = 1:size(PnL_scenario, 2)
  R_scenario(:, d) = PnL_scenario(:, d) / S(end, d);
end
num_alloc = 10; % number of allocations in between the min-variance and max-return allocations.
% Generate the mean-variance efficient frontier of returns under the posterior p_.
% - r: the vector of returns on the efficient frontier.
%   r(1) is the return of min-variance allocation.
%   r(end) is the return of max-return allocation.
% - vol: the vector of volatilities on the efficient frontier.
%   vol(1) is the volatility of min-variance allocation.
%   vol(end) is the volatility of max-return allocation.
% - W: the Nx(num_alloc+2) weight matrix, where each column defines an allocation
%   on the mean-variance frontier.
[r, vol, W, meanR, covR] = mv_frontier(R_scenario, p_, num_alloc);
% Save the efficient frontier.
if use_data_small == 1
  save('mv_frontier_small.mat', 'r', 'vol', 'W', 'meanR', 'covR');
else
  save('mv_frontier.mat', 'r', 'vol', 'W', 'meanR', 'covR');
end

% Plot the mean-variance efficient frontier.
figure;
plot(vol, r, '-x', 'MarkerFaceColor', 'r', 'MarkerSize', 5, 'LineWidth', 2);
grid on;
set(gca, 'xlim', [min(vol) max(vol)]);
xlabel('volatility', 'FontSize', 20);
ylabel('return', 'FontSize', 20);
set(gca, 'fontsize', 20);
saveas(gcf, 'mv_frontier.jpg');

% In order to pick one allocation from all those on the efficient frontier, we
% use CVaR of portfolio P&L under p_ as the index of satisfaction.
% Assume the total wealth is $1,000,000 dollars.
total_wealth = 1e6;
conf = 0.95; % confidence level.
CVaR_vec = zeros(1, size(W, 2));
for i = 1:size(W, 2)
  % The scenarios of portfolio P&L achieved by w. PnL_w should be a column vector.
  PnL_w = R_scenario * (total_wealth * W(:, i));
  [sort_PnL_w, idx] = sort(PnL_w);
  sort_p_ = p_(idx);
  % Force sort_p_ to be a row vector.
  if size(sort_p_, 1) > size(sort_p_, 2)
    sort_p_ = sort_p_';
  end
  % Find the cut at (1-conf) probability mass. 
  cut = sum(cumsum(sort_p_) <= (1-conf));
  CVaR_vec(i) = -(sort_p_(1:cut) * sort_PnL_w(1:cut)) / sum(sort_p_(1:cut));
end
% Find the minCVaR and the index.
[minCVaR, min_index] = min(CVaR_vec);
% Display the final allocation decision.
display('The minimum CVaR achieved on the efficient frontier is:');
disp(minCVaR);
display('with the return of:');
disp(r(min_index));
display('and volatility of:');
disp(vol(min_index));
