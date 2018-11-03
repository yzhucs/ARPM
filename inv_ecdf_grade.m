function x = inv_ecdf_grade(data, grade)
% function x = inv_ecdf_grade(data, grade)
% parameters:
% - data: a vector of data defining the empirical distribution.
% - grade: the vector of grades we want to invert to find the locations.
% returns:
% - x: a vector of locations as the inverses to the grade.
%
% Author: Yao Zhu (yzhucs@gmail.com)

x = zeros(size(grade));
[f, data_sort] = ecdf(data);
n = length(data_sort);
for i = 1:length(grade)
  % We define x(i) to be the data point such that its grade is no less than the given grade(i).
  index = sum(f < grade(i)) + 1;
  x(i) = data_sort(min(index, n));
end

return;