import os
import matplotlib.pyplot as plt

parent_folder_path = os.path.expanduser("~/Git_repo/flowstar_tree_code/mpc-verification/Automation_dryrun_3/cart_pendulum/main")

iteration_numbers = []
test_losses = []

# Iterate through each iteration folder
for i in range(13):
    iteration_folder_path = os.path.join(parent_folder_path, "Iteration_{}".format(i), "output_NN_training")
    output_file_path = os.path.join(iteration_folder_path, "_outputfile.txt")

    # Check if the output file exists in the iteration folder
    if os.path.isfile(output_file_path):
        with open(output_file_path, "r") as file:
            # Read the file contents
            file_contents = file.read()

            # Search for the specific string
            start_index = file_contents.find("The test loss is: ")
            if start_index != -1:
                start_index += len("The test loss is: ")
                end_index = file_contents.find("\n", start_index)
                if end_index != -1:
                    test_loss = float(file_contents[start_index:end_index])

                    # Store the iteration number and test loss
                    iteration_numbers.append(i)
                    test_losses.append(test_loss)

                    print("Iteration: {}, Test Loss: {}".format(i, test_loss))

# Plotting the line graph
plt.plot(iteration_numbers, test_losses, marker='o')
plt.xlabel("DNN")
plt.ylabel("Test Loss")
plt.title("Test Loss for each DNN against its own test set")
plt.xticks(range(13), ["DNN{}".format(i) for i in range(13)])
plt.grid(True)

# Save the plot as a figure (e.g., in PNG format)
plt.savefig("lineplot.png")

# Display a message indicating the figure has been saved
print("Line plot saved as lineplot.png")

