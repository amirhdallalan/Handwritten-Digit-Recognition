function CheckNNGradients(lambda)

    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0;
    end
    
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;
    
    % generate some random test data
    theta1 = DebugInitializeWeights(hidden_layer_size, input_layer_size);
    theta2 = DebugInitializeWeights(num_labels, hidden_layer_size);
    
    X = DebugInitializeWeights(m, input_layer_size - 1);
    y = 1 + mod(1:m, num_labels)';
    
    nn_params = [theta1(:); theta2(:)];
    
    CostFunc = @(p) NNCostFunction(p, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, lambda);
    [cost, grad] = CostFunc(nn_params);
    numgrad = ComputeNumericalGradient(CostFunc, nn_params);
    
    disp([numgrad grad]);
    fprintf('Left: Numerical Gradient | Right: Analytical Gradient\n');
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    fprintf('Relative Diffrence %g \n', diff);
end
     
     
     