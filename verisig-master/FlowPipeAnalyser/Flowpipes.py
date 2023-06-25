import json

from TaylorModel import TaylorModel

class FlowPipe:

    """
    POD Flowpipe class, contains a dictionary of variable names from the hybrid automata to their value. As well
    as some meta data.
    ---
    statevars: a dictionary from variables to their respective taylormodel
    branchID: the branch id in flowstar (-1 if unknown)
    modename: name of the mode for which the fp is given
    modeID: the id of the mode
    jumpsexecuted: number of jumps taken before arriving at this node.
    """

    def __init__(self):
        self.statevars = {}
        self.branchID = -1
        self.modeName = ""
        self.modeID = -1
        self.jumpsexecuted = -1
        self.domains = []
        self.unsafe = False

    def __str__(self):

        out = ""
        out += self.modeName + "\t" + "modeID:" + str(self.modeID) +"\t" + "branchID:" + str(self.branchID) +"\t" + "jumps executed:" + str(self.jumpsexecuted) + "\t" + "unsafe:" + str(self.unsafe) + "\n"
        for var in self.statevars.keys():
            out += str(var) + ": " + str(self.statevars[var])
            out += "\n"
        out += str(self.domains)
        return out

    def from_json(self, jsonnode):
        self.modeName = jsonnode["mode"]
        self.branchID = int(jsonnode["branchID"])
        self.modeID = int(jsonnode["modeID"])
        self.jumpsexecuted = int(jsonnode["jumpsExecuted"])
        self.unsafe = bool(jsonnode["unsafe"])
        if "flowpipe" in jsonnode.keys():
            fp = jsonnode["flowpipe"]
            for key in fp.keys():
                tm = TaylorModel()
                tm.from_json(fp[key])
                self.statevars[key] = tm
        if "domains" in jsonnode.keys():
            self.domains = jsonnode["domains"]
            pass

    def get_interval(self):
        """returns the intervals of all variables as a dictionary varname -> interval"""
        out = {}
        for key in self.statevars.keys():
            out[key] = self.statevars[key].get_interval(self.domains)
        return out

    def _get_interval(self, varname):
        """returns the interval of a given variable. This function should be considered private."""
        return self.statevars[varname].get_interval(self.domains)






class FlowpipeList:
    """
    Basic POD class for a list of flowpipes.
    --
    flowpipes: a list of flowpipes
    flowpipenames: a list of the keys in the original json, can be useful to check creation order in flowstar.
    """
    def __init__(self):
        self.flowpipes = []
        self.flowpipenames = []

    def from_json(self, jsonnode):
        '''Reads a list of flowpipes from a given json node.'''
        for key in jsonnode.keys():
            #next line necessary, since tranitions is at same depth.
            if key != "transitions":
                flowpipe = FlowPipe()
                flowpipe.from_json(jsonnode[key])
                self.flowpipes.append(flowpipe)
                self.flowpipenames.append(key)

    def __str__(self):
        '''returns a string representation of the Flowpipe list.'''
        out = ""
        for i in range(len(self.flowpipes)):
            out += self.flowpipenames[i]
            out += "\n"
            out += str(self.flowpipes[i])
            out += "\n---\n"
        return out

    def get_counter_example(self):
        '''Returns the counter example as a tuple (mode id, branch id, jumpsexecuted). If none is found, (-1, -1, -1)
        is returned'''
        for fp in self.flowpipes:
            if fp.unsafe :
                return (fp.modeID, fp.branchID, fp.jumpsexecuted)
        return (-1, -1, -1)

    def __len__(self):
        '''returns the length of the list of flowpipes.'''
        return len(self.flowpipes)

    def prune(self, mask):
        '''Given a mask, which is a set of tuples( mode id, branch id, jumpsexecuted) returns a new flowpipe
        list with only the flowpipes of these elements.'''
        out = FlowpipeList()
        for i in range(len(self.flowpipes)):
            fp = self.flowpipes[i]
            for m in mask:
                if fp.modeID == m.modeID and fp.branchID == m.branchID and fp.jumpsexecuted == m.jumpsExecuted:
                    out.flowpipes.append(fp)
                    out.flowpipenames.append(self.flowpipenames[i])
        assert(len(out) == len(mask))
        return out





if __name__ == '__main__':
    fp = FlowPipe()

    examplefile = open('Testfiles/Flowpipes.json', 'r')
    jsonex = json.load(examplefile)


    fpl = FlowpipeList()
    fpl.from_json(jsonex)

    print(fpl.get_counter_example())

    #print(str(fpl))

    fp = fpl.flowpipes[5]

    print(fp.get_interval())
#    print( fp._get_interval("clock") )
