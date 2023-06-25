# This script reformats the counterexample file for easier parsing

import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: python Extract.py <input_file> <output_file>")
    sys.exit(1)

# Get the input and output file names from the command-line arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# Open the input text file
with open(input_file, 'r') as file:
    # Read the file contents
    content = file.read()

# Split the content into individual lines
lines = content.split('\n')

# Initialize a flag to check if a set is being read
reading_set = False

# Initialize a variable to store the current set
current_set = ''

# Initialize a list to store sets containing "environment"
sets_with_environment = []

# Iterate over the lines
for line in lines:
    # Check if the line starts with 'environment'
    if line.startswith('environment'):
        # Set the flag to indicate that a set is being read
        reading_set = True

    # If a set is being read, append the line to the current set
    if reading_set:
        current_set += line + '\n'

    # Check if the line ends with '}' to determine the end of a set
    if line.endswith('}'):
        # Reset the flag and add the current set to the list if it contains "environment"
        reading_set = False
        if 'environment' in current_set:
            sets_with_environment.append(current_set)

        # Reset the current set variable for the next set
        current_set = ''

# Output the sets containing "environment" to the output text file
with open(output_file, 'w') as output:
    for set_string in sets_with_environment:
        output.write(set_string + '\n')

