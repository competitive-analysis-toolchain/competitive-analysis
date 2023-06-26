import os
import matplotlib.pyplot as plt

parent_folder_path = os.path.expanduser("~/Git_repo/flowstar_tree_code/mpc-verification/Automation_dryrun_3/motion_planning/main")

iteration_numbers = []
test_losses = []

# Iterate through each iteration folder
#for i in range(3):
for i in range(21):
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

# Set the figure size
fig, ax = plt.subplots(figsize=(8, 8))

# Plotting the line graph
plt.plot(iteration_numbers, test_losses, marker='o')
plt.xlabel("DNN")
plt.ylabel("Test Loss")
plt.title("Test Loss for each DNN against its own test set")
#plt.xticks(range(3), ["DNN{}".format(i) for i in range(3)])
plt.xticks(range(21), ["DNN{}".format(i) for i in range(21)])
plt.grid(True)

# Rotate y-axis labels for better spacing
plt.xticks(rotation=90)

# Save the plot as a figure
plt.savefig("lineplot.pdf")
plt.savefig("lineplot.png")
plt.savefig("lineplot.eps")

# Display a message indicating the plot has been saved
print("Line plot saved")

