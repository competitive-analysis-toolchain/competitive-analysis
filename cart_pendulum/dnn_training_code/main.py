from argparse import ArgumentParser
from argparse import ArgumentTypeError
from collections import defaultdict, deque
from general_parser import retrain_for_cegar

from decimal import Decimal
from itertools import chain
import json
import csv
import sys
from Use_cases.Cart_pole.cart_pole_use_case import Cart_pole_Use_case
from Use_cases.Motion_planning.motion_planning_use_case import Motion_planning_Use_case

def parse_args():
    parser = ArgumentParser()
    # set up argument parser to parse arguments (`python main.py --help` to view message)
    parser.add_argument("--flow-file",default="autosig_1.flow",help="input file")
    parser.add_argument("--controller-file",default="controller.csv", help="controller input file")
    parser.add_argument("--input-variables",default=["posD","thetaD","dposD","dthetaD"],help="input variables",nargs="+")
    parser.add_argument("--output-variables",default=["FD"],help="input variables",nargs="+")
    parser.add_argument("--amount_interval_points",default=4,help="number of extra points between low and high",type=int)
    parser.add_argument("--decimal-precision",default=6,help="number of decimal places in the output",type=int)
    parser.add_argument("--output-file",default="retraining_data.csv", help="output file")

    parser.add_argument("--old-dataset",default="data_set_mpc_example.csv",help="old dataset")
    parser.add_argument("--hyper-training-epochs",default=200,help="number of epochs used as an argument for hypertraining",type=int)
    parser.add_argument("--hyper-training-factor",default=3,help="number of epochs used as an argument for hypertraining",type=int)
    parser.add_argument("--NN-training-epochs",default=400,help="amount of epochs that the NN is trained",type=int)
    parser.add_argument("--output-folder",default="retraining_output",help="folder where the results are trained")
    parser.add_argument("--hypertune",default="false",help="bool that says if you hypertune or not")
    parser.add_argument("--hyperparameter_file",default="hyperparameter_file",help="the file where the hyperparameters are saved when hypertuning and where the hyperparameters are read from when training")

    return parser.parse_args()

def to_decimal(value):
    return Decimal(value).quantize(Decimal("." + opt.decimal_precision * "0"))

def parse_flow_data(filename):
    variable_name = None # name of the current variable being parsed
    unparsed_data = "" # json data to parse
    flow_data = [] # non unique keys so retain dictionary
    with open(filename) as f:
        # make sure that the last line is empty
        for line in chain(f, [""]):
            line = line.strip()
            if not line:
                if unparsed_data and variable_name is not None:
                    #flow_data[-1] |= json.loads(unparsed_data.replace("\'", "\""))
                    flow_data[-1].update(json.loads(unparsed_data.replace("\'", "\"")))
                    # convert floats to decimals
                    #flow_data[-1] |= {k: [to_decimal(v[0]), to_decimal(v[1])] for k, v in flow_data[-1].items() if isinstance(v, list)}
                    flow_data[-1].update({k: [to_decimal(v[0]), to_decimal(v[1])] for k, v in flow_data[-1].items() if isinstance(v, list)})
                    unparsed_data = ""
                    variable_name = None
            # add to the JSON if
            # the line starts with {
            # there is currently data being parsed
            elif line.startswith("{") or unparsed_data:
                unparsed_data += line
            else:
                # start with the name
                name, unparsed_variables = line.split(maxsplit=1)
                variable_data = {"variable": name}
                # make sure line ends with a space (easier for parsing)
                unparsed_variables += " "
                while unparsed_variables:
                    # parse pairs separated by ": " and " "
                    variable, unparsed_variables = unparsed_variables.split(": ", maxsplit=1)
                    value, unparsed_variables = unparsed_variables.split(" ", maxsplit=1)
                    variable_data[variable] = int(value)
                flow_data.append(variable_data)
                variable_name = name
    return flow_data

def parse_csv_data(filename):
    with open(filename) as f:
        # reads data quickly but returns only strings
        reader = csv.DictReader(f, delimiter=",")
        return list(reader)

def filter_keys(filtered_keys, data):
    """keep only keys from data that are in `filtered_keys`"""
    ignored_keys = []
    for variable in data:
        if variable not in filtered_keys:
            ignored_keys.append(variable)
    for variable in ignored_keys:
        del data[variable]
    return data

def save_csv_data(csv_data, range_data, add_new_data):
    NEW_COLUMN = "new data/old data"
    for row in csv_data:
        # filter keys and add "old data" in the `NEW_COLUMN`
        filter_keys(range_data, row)
        row[NEW_COLUMN] = "old data"
    
    if add_new_data:
        for range_ in range_data.values():
            # add the median value
            range_.append((range_[0] + range_[1]) / 2)
        for i in range(3):
            csv_data.append({variable: range_[i] for variable, range_ in range_data.items()})
            csv_data[-1][NEW_COLUMN] = "new data"

    # put the new column at the start
    fieldnames = list(csv_data[-1].keys())
    fieldnames.insert(0, fieldnames.pop(-1))

    with open(opt.output_file, "w", newline="") as f:
        # write CSV to file
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

def process_data(flow_data, csv_data):
    """complete checks and save output file"""
    # sort by jumpsexecuted
    variables = sorted(flow_data, key=lambda data: data["jumpsexecuted"])
    flow_data = iter(variables)

    for line in csv_data:
        # convert to decimal
        for k, v in line.items():
            if k in opt.input_variables or k in opt.output_variables:
                # convert any non-integers to floats
                line[k] = to_decimal(v)
        # easier to work with
        line["index"] = int(line["initial position index"])
        line["timestep"] = int(line["timestep"])

    position = None
    successful_comparison = False
    for line in sorted(csv_data, key=lambda data: (data["index"], data["timestep"])):
        if not successful_comparison or line["index"] != position:
            # reset if last line has failed
            # reset when the index changes
            position = line["index"]
            timesteps = []
            initial_positions = []
            flow_data = iter(variables)
            range_data = next(flow_data)

        # check each line
        successful_comparison = check_line(line, range_data)
        if successful_comparison:
            timesteps.append(line["timestep"])
            initial_positions.append(line["index"])
            try:
                range_data = next(flow_data)
            except StopIteration:
                # completed the last check successfully
                break
    else: 
        return save_csv_data(
            # include only the previous successful lines
            csv_data,
            {
                # use only ranges that appear in the input
                var: range_data[var]
                for var in filter_keys(
                    opt.input_variables,
                    line
                )
            },
            add_new_data=True
        )

    # all comparisons are good as we have not returned yet
    print("real counterexample")
    print("initial position index:", ",".join(map(str, initial_positions)))
    print("timestep:", ",".join(map(str, timesteps)))
    # save csv but don't append (range_data is None)
    save_csv_data(csv_data, range_data, add_new_data=False)

def check_line(line, range_data):
    for variable in opt.input_variables + opt.output_variables:
        if not variable in range_data:
            # only check valid ranges
            continue
        value = line[variable]
        range = range_data[variable]
        # check if variable is in range
        if not (range[0] <= value and value <= range[1]):
            return False
    # return true only if all values on the line match
    return True

def main(opt):
    flow_data = parse_flow_data(opt.flow_file)
    csv_data = parse_csv_data(opt.controller_file)
    process_data(flow_data, csv_data)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    use_case = Cart_pole_Use_case()
    use_case.set_self_parameters()
    opt = parse_args()
    main(opt)
    print(opt.hypertune)
    retrain_for_cegar(use_case, opt.output_file, len(opt.input_variables), len(opt.output_variables), opt.amount_interval_points, opt.old_dataset, opt.hyper_training_epochs, opt.hyper_training_factor, opt.NN_training_epochs, opt.output_folder, str2bool(opt.hypertune), opt.hyperparameter_file)
