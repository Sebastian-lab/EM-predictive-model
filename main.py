 spect_data = []
with open('spectX.txt') as file:
    for line in file.readlines():
        features = []
        for col in line.strip():
            if col != ' ':
                features.append(int(col))
        spect_data.append(features)

with open('spectY.txt') as file:
    labels = [int(line) for line in file.read().splitlines()]

import numpy as np

num_examples = 267  # examples
num_features = 23  # features
X_data = np.array(spect_data)  # (num_examples, num_features)
Y_labels = np.array(labels)  # (num_examples)
probabilities = np.full(num_features, 0.05)
log_table = []  # iteration, num of mistakes, log-likelihood

# EM algorithm
for iteration in range(257):
    # E-Step
    expected_latent = np.zeros((num_examples, num_features))
    for example_idx in range(num_examples):
        denominator = 1 - np.prod([(1 - probabilities[feature_idx]) ** X_data[example_idx][feature_idx] for feature_idx in range(num_features)])
        for feature_idx in range(num_features):
            expected_latent[example_idx, feature_idx] = (Y_labels[example_idx] * X_data[example_idx][feature_idx] * probabilities[feature_idx]) / denominator

    # M-Step
    new_probabilities = np.zeros(num_features)
    for feature_idx in range(num_features):
        feature_sum = np.sum(X_data[:, feature_idx])
        new_probabilities[feature_idx] = np.sum(expected_latent[:, feature_idx]) / feature_sum if feature_sum > 0 else probabilities[feature_idx]

    # Log-likelihood
    log_likelihood = 0
    for example_idx in range(num_examples):
        # P(Y = 0 | X = x)
        prob_y_given_x = np.prod([(1 - probabilities[feature_idx]) ** X_data[example_idx][feature_idx] for feature_idx in range(num_features)])
        if Y_labels[example_idx] == 1:  # P(Y = 1 | X = x)
            prob_y_given_x = 1 - prob_y_given_x
        log_likelihood += np.log(prob_y_given_x)
    log_likelihood /= num_examples

    # Number of mistakes
    num_mistakes = 0
    for example_idx in range(num_examples):
        # P(Y = 1 | X = x)
        prob_y_given_x = 1 - np.prod([(1 - probabilities[feature_idx]) ** X_data[example_idx][feature_idx] for feature_idx in range(num_features)])

        # false positive or false negative
        if ((Y_labels[example_idx] == 0 and prob_y_given_x >= 0.5) or
                (Y_labels[example_idx] == 1 and prob_y_given_x <= 0.5)):
            num_mistakes += 1

    # Append table to fill out
    log_table.append([iteration, num_mistakes, log_likelihood])

    # Update probabilities with new values
    probabilities = new_probabilities

log_table = np.array(log_table).reshape(-1, 3)
print(log_table)
