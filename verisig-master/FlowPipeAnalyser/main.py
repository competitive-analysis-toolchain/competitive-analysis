from argparse import ArgumentParser
from Flowpipes import FlowpipeList
from Traces import TraceTree, build_trace_tree,build_trace_tree_depth

import json


if __name__ == '__main__':

    ap = ArgumentParser(description='parser of arguments')

    ap.add_argument("-json", action="store", dest="inputfile", help="The json file that will be parsed.", required=True)
    ap.add_argument("-o", action="store", dest="outputfile", help="The output file that will be used.")

    args = ap.parse_args()

    #load json into memory
    examplefile = open(args.inputfile, 'r')
    jsonex = json.load(examplefile)
    fpl = FlowpipeList()
    fpl.from_json(jsonex)

    #get path
    counterexample = fpl.get_counter_example()
    if counterexample == (-1, -1, -1):
        print("No counter example present. Aborting.")
        exit(0)


    transitions = jsonex["transitions"]
    initmid = jsonex["flowpipe0"]["modeID"]
    tree = build_trace_tree_depth(transitions, initmid, counterexample[2])

    leaf = tree.get_node(counterexample[0], counterexample[1], counterexample[2])
    path = leaf.get_path()

    #prune the flowpipelist
    fpl_pruned = fpl.prune(path)

    for fp in fpl_pruned.flowpipes:
        print(fp.modeName,  "branchid:", fp.branchID, "jumpsexecuted:", fp.jumpsexecuted)
        int = fp.get_interval()
        print(int)













