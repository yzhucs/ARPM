function grade = ecdf_grade(data, x)
% function grade = ecdf_grade(data, x)
% parameters:
% - data: a vector of data defining the empirical distribution.
% - x: a vector of locations to evaluate the grade defined by the empirical cdf.
% returns:
% - grade: the vector of grades corresponding to x.
%
% Author: Yao Zhu (yzhucs@gmail.com)

grade = zeros(size(x));
n = length(data);
for i = 1:length(x)
  grade(i) = sum(data <= x(i)) / n;
end

return;