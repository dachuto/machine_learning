function plotFit(min_x, max_x, mu, sigma, theta, p, color)
%PLOTFIT Plots a learned polynomial regression fit over an existing figure.
%Also works with linear regression.
%   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
%   fit with power p and feature normalization (mu, sigma).


% We plot a range slightly bigger than the min and max values to get
% an idea of how the fit will vary outside the range of the data points
border = (max_x - min_x) * 0.2;
x = (min_x - border: 0.05 : max_x + border)';

% Map the X values 
X_poly = polyFeatures(x, p);
X_poly = bsxfun(@minus, X_poly, mu);
X_poly = bsxfun(@rdivide, X_poly, sigma);

% Add ones
X_poly = [ones(size(x, 1), 1) X_poly];

% Plot
plot(x, X_poly * theta, '--', 'LineWidth', 2, 'color', color)

end
