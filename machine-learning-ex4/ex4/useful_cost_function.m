function J = useful_cost_function(hx, y)
J = log(hx) .* -y - log(1 - hx) .* (1 - y);
end
