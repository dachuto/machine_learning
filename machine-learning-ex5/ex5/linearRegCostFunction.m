function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = X * theta;
diff = h - y;

theta_no_zero = [0; theta(2:end, :)];
J = (sum(diff .^ 2) + lambda * sum(theta_no_zero' * theta_no_zero)) / (2 * m);

grad = (X' * diff + lambda * theta_no_zero) / m;
end
