%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

train = csvread('train.csv');
X = train(:, [2:end]); y = train(:, 1);

val = csvread('validation.csv');
Xval = val(:, [2:end]); yval = val(:, 1);

fprintf('\nProgram paused #1. Press enter to continue.\n');
pause;

%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunctionReg.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
lambda = 1;
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
%plotDecisionBoundary(theta, X, y);

% Put some labels 
%hold on;
% Labels and Legend
%xlabel('Exam 1 score')
%ylabel('Exam 2 score')

% Specified in plot order
%legend('Admitted', 'Not admitted')
%hold off;

fprintf('\nProgram paused. Press enter to continue. Next is Train Accuracy \n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

%prob = sigmoid([1 45 85] * theta);
%fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
        % 'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%

% ~29000 training examples is too many iterate on!!!
% Just use 1000 increments up to 10,000, 10 buckets
buckets = 10;

lambda = 1;
[error_train, error_val] = ...
    learningCurve(X, y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:buckets, error_train, 1:buckets, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([1 10 0 1])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:buckets
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ========= Part 6: Precision & Recall =============
%% The learning curve is very flat. The outcome variable is skewed.
%% Need precision and recall to tell if the alogithm is improving or nor

true_positives = 0;
false_positives = 0;

true_negative = 0;
false_negative = 0;

for i = 1:m

	if p(i) == 1
		if p(i) == y(i)
			true_positives = true_positives + 1;
		else
			false_positives = false_positives + 1;
		end
	else
		if p(i) == y(i)
			true_negative = true_negative + 1;
		else
			false_negative = false_negative +1;

		end
	end
end
fprintf('True Positives: %f\n', true_positives);
fprintf('False Positives: %f\n', false_positives);

fprintf('True Negative: %f\n', true_negative);
fprintf('False Negative: %f\n', false_negative);

%% Precision: True Positives / # predicted positives
%%            True Positives / (true positives + false positives)

precision = (true_positives / (true_positives + false_positives));
fprintf('Precision: %f\n', precision);

%% Recall: True Positives / # actual positives
%%         True Positives / (true pos + false neg)

recall = (true_positives / (true_positives + false_negative));
fprintf('Recall: %f\n', recall);


F_score = ((2 * precision * recall) / (precision + recall));
fprintf('F Score: %f\n', F_score);


val_pred = predict(theta, [ones(size(Xval, 1), 1) Xval]);

auc = scoreAUC(yval, val_pred);
fprintf('AUC: %f\n', auc);

%% ============== Compare on Test and Submit =================


load('test.mat');


Xtest = double(test(:,2:end)); id = double(test(:,1));

output = predict(theta, [ones(size(Xtest, 1), 1) Xtest]);

results = [id, output];
csvwrite('final', results);

