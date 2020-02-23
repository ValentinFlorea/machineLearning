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
Theta1Reshaped = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2Reshaped = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1Reshaped));
Theta2_grad = zeros(size(Theta2Reshaped));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



X_ones = [ones(m, 1) X];
z2 = X_ones * Theta1Reshaped';
a2 =  [ones(m, 1) sigmoid(z2)];
hTheta = sigmoid(a2 * Theta2Reshaped');

%J = sum((-y' * log(sigmoidResult)) - (1-y') * log((1-sigmoidResult)))/m ;
%+ sum(Theta1Reshaped(2:end).^2)*lambda/(2*m);

Y_ones = zeros(m, num_labels);
for i = 1:m 
Y_ones(i, y(i)) = 1;
end;

J = sum(sum((-Y_ones .* log(hTheta) - (1 - Y_ones) .* log(1 - hTheta)))) / m;
regularizationJ =  (sum(sum(Theta1Reshaped(:,2:end) .^ 2)) + sum(sum(Theta2Reshaped(:,2:end) .^ 2))) * lambda / (2 * m);
J = J + regularizationJ;



delta3 = hTheta - Y_ones;
delta2 = delta3 * Theta2Reshaped .* (a2 .* (1 - a2));

Delta1 = zeros(size(Theta1Reshaped));
Delta2 = zeros(size(Theta2Reshaped));

Delta1 = Delta1 +  delta2(:, 2:end)' * X_ones;
Delta2 = Delta2 +  delta3' * a2;

%Theta1_grad = (Delta1 + (lambda * Theta1Reshaped .*
%[zeros(size(Theta1Reshaped,1), 1) ones(size(Theta1Reshaped,1),
%size(Theta1Reshaped, 2) -1)]))/m; % works but very comples
%Theta2_grad = (Delta2 + (lambda * Theta2Reshaped .*
%[zeros(size(Theta2Reshaped,1), 1) ones(size(Theta2Reshaped,1),
%size(Theta2Reshaped, 2) -1)]))/m; % works but very complex


ZeroColumnTheta1 = Theta1Reshaped;
ZeroColumnTheta2 = Theta2Reshaped;
ZeroColumnTheta1(:,1) = 0;
ZeroColumnTheta2(:,1) = 0;


Theta1_grad = (Delta1 + (lambda * ZeroColumnTheta1))/m;
Theta2_grad = (Delta2 + (lambda * ZeroColumnTheta2))/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
