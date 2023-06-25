
class TraceTree:
    '''
    Class for a trace tree (node)
    ---
    each node contains the mode id, branch id, and jumps executed that corresponds to a step in the execution trace.
    Additionally each node contains its parent and children.
    '''

    def __init__(self, mid = 1, bid = 1, jex = 1):
        self.modeID  = mid
        self.branchID = bid
        self.jumpsExecuted = jex
        self.parent = None
        self.children = []


    def print_tree(self, depth):
        '''prints the tree in terminal. '''
        print( " " * depth , "(", self.modeID, ", ", self.branchID, ", ", self.jumpsExecuted, ")")
        for c in self.children:
            c.print_tree(depth + 1)

    def to_dot(self, root=True):
        '''returns the dot representation of a tracetree.
        Root should be set to true when calling  the function to indicate that is the root.'''
        out = ""
        if root:
            out += "digraph tracetree { \n"
        label = "(" + str(self.modeID) + "," + str(self.branchID) + ", " + str(self.jumpsExecuted) + ")"
        id = str(self.modeID) + str(self.branchID) + str(self.jumpsExecuted)
        out += "c_" + id + "[label=\"" + label + "\"]; \n"

        for c in self.children:
            out += c.to_dot(False)

            idc = str(c.modeID) + str(c.branchID) + str(c.jumpsExecuted)
            out += "c_" + id + "->c_" + idc + ";\n"

        if root:
            out += "}\n"
        return out

    def get_node(self, mid, bid, jex):
        '''returns a specific node with the given mode id, branch id, and jumps executed.'''
        #find correct node
        fringe = [self]
        while True:
            if len(fringe) == 0:
                return None
            node = fringe[0]
            del fringe[0]
            if node.modeID == mid and node.branchID == bid and node.jumpsExecuted == jex:
                return node
            else:
                fringe += node.children

    def get_path(self):
        '''Returns the path from the given node to its root.'''
        out = [self]
        node = self
        while node.parent:
            out += [node.parent]
            node = node.parent
        return out


def build_trace_tree_depth(transitions, initmodeid, depth):
    """build the trace tree up to the target (modeid, branchid, jumpsexecuted)"""
    #build transition dict
    transdict = to_trans_dict(transitions)

    #make root
    root = TraceTree(initmodeid, 1, 0)

    fringe = [root]
    index = 1
    while True:
       # root.print_tree(0)
        node = fringe[0]
        del fringe[0]

        # base case
        if node.jumpsExecuted > depth:
            return root

        if len(transdict[node.modeID]) <= 0:
            continue

        else:
            childids = transdict[node.modeID]

            newnode = TraceTree(childids[0], node.branchID, node.jumpsExecuted + 1)
            newnode.parent = node
            childs = [newnode]
            for i in range(1, len(childids)):
                index += 1
                newnode = TraceTree(childids[i], index, node.jumpsExecuted + 1)
                newnode.parent = node
                childs.append(newnode)
            node.children = childs
            fringe += childs

def build_trace_tree(transitions, initmodeid, target):
    """build the trace tree up to the target (modeid, branchid, jumpsexecuted)"""
    #build transition dict
    transdict = to_trans_dict(transitions)

    #make root
    root = TraceTree(initmodeid, 1, 0)

    fringe = [root]
    index = 1
    while True:
        #root.print_tree(0)
        node = fringe[0]
        del fringe[0]

        # base case
        if node.modeID == target[0] and node.branchID == target[1] and node.jumpsExecuted == target[2]:
            return root

        if len(transdict[node.modeID]) <= 0:
            continue

        else:
            childids = transdict[node.modeID]

            newnode = TraceTree(childids[0], node.branchID, node.jumpsExecuted + 1)
            newnode.parent = node
            childs = [newnode]
            for i in range(1, len(childids)):
                index += 1
                newnode = TraceTree(childids[i], index, node.jumpsExecuted + 1)
                newnode.parent = node
                childs.append(newnode)
            node.children = childs
            fringe += childs


def to_trans_dict(transitions):
    """builds a transition dictionary for ease of use"""
    states = set([x[0] for x in transitions] + [x[1] for x in transitions])
    transdict = dict()
    for state in states:
        transdict[state] = [x[1] for x in transitions if x[0] == state]
    return transdict


def in_path(modeid, branchid, jumpsex, path):
    for node in path:
        if (modeid == node.modeID and branchid == node.branchID) and jumpsex == node.jumpsExecuted:
            return True
    return False


if __name__ == '__main__':
    t = [[0, 1], [0, 2], [1,2], [1,2], [2, 0]]

    r = build_trace_tree(t, 0,  ( 0 ,  7 ,  8 ))

    f = open("tracetree.dot", 'w')
    f.write(r.to_dot())
    f.close()


    target = r.get_node(0 ,  7 ,  8 )


