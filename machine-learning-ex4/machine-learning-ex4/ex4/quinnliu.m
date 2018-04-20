X = [ones(m, 1) X];
a1 = X;

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

a3 = sigmoid(a2 * Theta2');

ry = eye(num_labels)(y, :);

cost = ry .* log(a3) + (1 - ry) .* log(1 - a3);

J = -sum( sum(cost, 2) ) / m;

reg = sum( sum(Theta1(:, 2:end).^2) ) + sum( sum(Theta2(: , 2:end).^2));

J = J + lambda / (2 * m) * reg;

G1 = zeros( size(Theta1) );
G2 = zeros( size(Theta2) );

for i = 1:m,
	ra1 = X(i, :)';

	rz2 = Theta1 * ra1;
	ra2 = sigmoid(rz2);
	ra2 = [1; ra2];

	rz3 = Theta2 * ra2;
	ra3 = sigmoid(rz3);

	err3 = ra3 - ry(i, :)';

	err2 = (Theta2' * err3)(2:end, 1) .* sigmoidGradient(rz2);

	G1 = G1 + err2 * ra1';
	G2 = G2 + err3 * ra2';
end

Theta1_grad = G1 / m + lambda * [zeros(hidden_layer_size, 1) Theta1(:, 2:end)] / m;
Theta2_grad = G2 / m + lambda * [zeros(num_labels, 1) Theta2(:, 2:end)] / m;