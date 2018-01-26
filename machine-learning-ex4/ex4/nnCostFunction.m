function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
	Y(i, :)= I(y(i), :);
end

% feed forward
a_1 = expand_ones(X);
z_2 = a_1 * Theta1';
hidden_layer = sigmoid(z_2);
a_2 = expand_ones(hidden_layer);
output_layer = sigmoid(a_2 * Theta2');

J = useful_cost_function(output_layer, Y);
J = sum(J(:)) / m;

regularization_cost = lambda * 0.5 / m * (sum(sum(cut_first_column(Theta1) .^2, 2)) + sum(sum(cut_first_column(Theta2).^2, 2)));
J = J + regularization_cost;

sigma_3 = output_layer - Y;
sigma_2 = cut_first_column(sigma_3 * Theta2 .* sigmoidGradient(expand_ones(z_2)));

delta_1 = sigma_2' * a_1;
delta_2 = sigma_3' * a_2;

Theta1_grad = (delta_1 ./ m) + (lambda / m) * [zeros(size(Theta1,1), 1) cut_first_column(Theta1)];
Theta2_grad = (delta_2 ./ m) + (lambda / m) * [zeros(size(Theta2,1), 1) cut_first_column(Theta2)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
