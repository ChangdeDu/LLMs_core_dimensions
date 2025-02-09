function [chance_level, lower_bound, upper_bound] = compute_chance_level(y_pred, y_true)


function accuracy = calculate_accuracy(y_true, y_pred)
    accuracy = sum(y_true == y_pred) / length(y_true);
end

original_accuracy = calculate_accuracy(y_true, y_pred);

num_permutations = 1000;
permuted_accuracies = zeros(1, num_permutations);

for i = 1:num_permutations
    permuted_y_true = y_true(randperm(length(y_true)));
    permuted_accuracies(i) = 100*calculate_accuracy(permuted_y_true, y_pred);
end

% chance level
chance_level = mean(permuted_accuracies);

% chance level 95% CI
lower_bound = prctile(permuted_accuracies, 2.5);
upper_bound = prctile(permuted_accuracies, 97.5);

fprintf('Original Accuracy: %.4f\n', original_accuracy);
fprintf('Chance Level: %.4f\n', chance_level);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound, upper_bound);
end

