import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# DNN models
model_paths = ["Iteration_{}/output_NN_training/dnn_model".format(i) for i in range(13)]

# Test sets
test_set_paths = ["Iteration_{}/output_NN_training/test_data_set_iteration{}.csv".format(i, i) for i in range(13)]

# Test loss
error_loss_data = []

# Load all DNNs
models = [tf.keras.models.load_model(model_path) for model_path in model_paths]

#Split the test set into test features and test labels
for test_set_index, test_set_path in enumerate(test_set_paths):
    test_set = pd.read_csv(test_set_path)
    test_features = test_set.iloc[:, 0:4]
    test_labels = test_set.iloc[:, 4]

    # Compute the error loss for each DNN against all test sets
    for dnn_index, model in enumerate(models):
        error_loss = model.evaluate(test_features, test_labels, batch_size=32, verbose=0)
        data_point = (dnn_index, test_set_index, error_loss)  # (DNN index, Test set index, Error loss)
        error_loss_data.append(data_point)
        
# Print the test loss for each DNN against all test sets
for data_point in error_loss_data:
    print("DNN{}, Test Set {}: Error Loss = {}".format(data_point[0], data_point[1], data_point[2]))

# Separate error losses by DNN
dnn_error_losses = [[] for _ in range(len(models))]
for data_point in error_loss_data:
    dnn_index, _, error_loss = data_point
    dnn_error_losses[dnn_index].append(error_loss)

# Create box plot
plt.boxplot(dnn_error_losses)

# Set x-axis labels
plt.xticks(range(1, len(models) + 1), ["DNN{}".format(i) for i in range(len(models))])

# Set y-axis label
plt.ylabel("Error Loss")

# Set plot title
plt.title("Box Plot of Error Loss for DNNs")

# Save the figure as a file (e.g., in PNG format)
plt.savefig("boxplot.png")

# Display a message indicating the figure has been saved
print("Box plot saved as boxplot.png")
