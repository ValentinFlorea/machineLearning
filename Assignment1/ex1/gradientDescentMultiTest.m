clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 1;
num_iters = 10;

% Init Theta and Run Gradient Descent
theta = zeros(3, 1);

[thetaGD, J_history] = gradientDescentMulti(X, y, theta, 1, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', thetaGD);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

Xprediction = ([1650, 3] - mu)./sigma
price = [1 Xprediction] * thetaGD
