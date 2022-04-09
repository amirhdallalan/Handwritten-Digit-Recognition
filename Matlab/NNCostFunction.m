function [J grad] = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    
    theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    m = size(X, 1);
    
    X = [ones(m,1), X];

    a1 = X;
    z2 = a1 * theta1';
    a2 = Sigmoid(z2);
    a2 = [ones(size(a2,1),1), a2];

    z3 = a2 * theta2';
    a3 = Sigmoid(z3);

    h_x = a3;

    y_Vec = (1:num_labels)==y;

    J = (1/m) * sum(sum((-y_Vec.*log(h_x))-((1-y_Vec).*log(1-h_x))));

    A1 = X;

    Z2 = A1 * theta1';
    A2 = Sigmoid(Z2);
    A2 = [ones(size(A2,1),1), A2];

    Z3 = A2 * theta2';
    A3 = Sigmoid(Z3);

    y_Vec = (1:num_labels)==y;

    DELTA3 = A3 - y_Vec;
    DELTA2 = (DELTA3 * theta2) .* [ones(size(Z2,1),1) SigmoidGradient(Z2)];
    DELTA2 = DELTA2(:,2:end);

    theta1_grad = (1/m) * (DELTA2' * A1);
    theta2_grad = (1/m) * (DELTA3' * A2);

    reg_term = (lambda/(2*m)) * (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2)));

    J = J + reg_term;

    theta1_grad_reg_term = (lambda/m) * [zeros(size(theta1, 1), 1) theta1(:,2:end)];
    theta2_grad_reg_term = (lambda/m) * [zeros(size(theta2, 1), 1) theta2(:,2:end)];

    theta1_grad = theta1_grad + theta1_grad_reg_term;
    theta2_grad = theta2_grad + theta2_grad_reg_term;
    
    grad = [theta1_grad(:) ; theta2_grad(:)];
end