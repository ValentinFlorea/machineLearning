function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



sigmaVect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
CVect = sigmaVect;

% for i = 1:length(CVect)
%     for j = 1:length(sigmaVect)
%         model = svmTrain(X, y, CVect(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVect(j)));
%         predictionsY = svmPredict(model, Xval);
%         predictionsError(i, j) =  mean(double(predictionsY ~= yval));
%     end;
% end;
% 
% [colMin, rowIndex] = min(predictionsError);
% [absMin, colIndex] = min(colMin);
% C = CVect(rowIndex(colIndex));
% sigma = sigmaVect(colIndex);
C = 1;
sigma = 0.1;

% =========================================================================

end
