function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
m = length(y); % number of training examples

errors = X * theta - y;
J = 1 / (2 * m) * errors' * errors;

end