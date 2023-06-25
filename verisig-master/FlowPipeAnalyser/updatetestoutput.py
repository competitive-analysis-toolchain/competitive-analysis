input_file = 'testoutput.txt'
output_file = 'modified_output.txt'

modes = []
current_mode = None

with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('initD_initN') or line.startswith('_DNN1'):
            current_mode = None
            continue
        if line.startswith('environment'):
            if current_mode is not None:
                modes.append(current_mode)
            current_mode = line.strip()
        elif current_mode is not None:
            current_mode += '\n' + line.strip()

if current_mode is not None:
    modes.append(current_mode)

jumpsexecuted_values = []
environment_modes = []
for mode in modes:
    line_parts = mode.split()
    jumpsexecuted_value = int(line_parts[line_parts.index('jumpsexecuted:') + 1])
    if line_parts[0].startswith('environment'):
        environment_modes.append((mode, jumpsexecuted_value))
    jumpsexecuted_values.append(jumpsexecuted_value)

min_jumpsexecuted = min(jumpsexecuted_values)

with open(output_file, 'w') as f:
    for i, (mode, jumpsexecuted) in enumerate(environment_modes):
        jumpsexecuted = str(jumpsexecuted - min_jumpsexecuted)
        f.write(mode.replace('jumpsexecuted: ' + str(jumpsexecuted_values[i]), 'jumpsexecuted: ' + jumpsexecuted))
        if i < len(environment_modes) - 1:
            f.write('\n')

