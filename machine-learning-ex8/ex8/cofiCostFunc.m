function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

diff = R .* (X * Theta' - Y);
sum_squared = @(A) sum(sum(A .^ 2));

J = (sum_squared(diff) + lambda * (sum_squared(X)  + sum_squared(Theta))) * 0.5;
X_grad = diff * Theta + lambda * X;
Theta_grad = diff' * X + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end
