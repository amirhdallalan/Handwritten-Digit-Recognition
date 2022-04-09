clear
close all
clc

%% Loading and Visualizing Data

load('dataset.mat')
m = size(X, 1);
vis = randperm(size(X, 1));
vis = vis(1:100);

DisplayData(X(vis, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Initializing Parameters
input_layer_size  = 400;
hidden_layer_size = 25;
num_labels = 10; 

initial_theta1 = RandInitialWeights(input_layer_size, hidden_layer_size);
initial_theta2 = RandInitialWeights(hidden_layer_size, num_labels);

initial_nn_params  = [initial_theta1(:); initial_theta2(:)];

%% Implement Backpropagation

CheckNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Training Neural Network

options = optimset('MaxIter', 200);

lambda = 1;
CostFunction = @ (p) NNCostFunction(p, ...
                                    input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels, X, y, lambda);
[nn_params, cost] = fmincg(CostFunction, initial_nn_params, options);

theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Implement Predict

pred = Predict(theta1, theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% Test Model

rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    close all
    DisplayData(X(rp(i), :));

    pred = Predict(theta1, theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end