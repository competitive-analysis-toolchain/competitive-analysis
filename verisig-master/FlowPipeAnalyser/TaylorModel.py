import json

from itertools import product


def contains_zero(a, b):
    """
    Returns true if 0 is an element in the interval [a, b]
    """
    assert(a <= b)
    return True if a <= 0 and b >= 0 else 0


class Monomial:
    """
    POD Class for representing monomials.
    --
    coeff: the coefficient given as an interval. An interval is represented as a list with 2 values [inf, sup].
    degrees: a list of degrees.
    """

    def __init__(self):
        self.coeff = [0, 0]
        self.degrees = []

    def __str__(self):
        out = ""
        out += str(self.coeff)
        out += str(self.degrees)
        return out

    def get_interval(self, domains):
        """ Computes the interval of values of the monomial given a set of intervals (aka the domain)"""

        if len(domains) != len(self.degrees):
            raise RuntimeError("size of domains does not match size of degrees.")
        inf = float('inf')
        sup = float('-inf')

        #compute all coefficients for the term of the monomial, with their power.
        coeffs = [self.coeff]
        powers = [1]

        for index in range(len(self.degrees)):
            degree = self.degrees[index]

            if degree > 0:
                coeffs.append(domains[index])
                powers.append(degree)


        #creates a list of tuples where each tuple is a possible combination of the coefficients (index of)
        combinations = list(product([0, 1], repeat= len(coeffs)))

        #create all combinations of values and update inf and sup
        for comb in combinations:
            value = 1
            for i in range(len(comb)):
                value *= coeffs[i][comb[i]] ** powers[i]
            inf = min(value, inf)
            sup = max(value, sup)

        #whenever 0 is in one of the intervals, it can be a bound
        for c in coeffs:
            if contains_zero(c[0], c[1]):
                inf = min(0, inf)
                sup = max(0, sup)
                break
        return [inf, sup]


class TaylorModel:
    """
    POD Class for representing Taylormodels
    --
    expansion: a list of monomials
    remainder: an interval with the remainder of the tm. An interval is represented as a list with 2 values [inf, sup].
    varnames: a list of names for the local variables, can be left empty
    """

    def __init__(self):
        self.expansion = []
        self.remainder = [0, 0]
        self.varnames = []

    def from_json(self, jsonnode):
        '''reads a taylormode from json. jsonnode needs to have keys coefficients, degrees, and remainder available in root.'''

        coeff = jsonnode["coefficients"]
        degs = jsonnode["degrees"]
        rem = jsonnode["remainder"]

        for i in range(len(coeff)):
            mono = Monomial()
            mono.coeff = coeff[i]
            mono.degrees = degs[i]

            self.expansion.append(mono)
        self.remainder = rem

    def get_interval(self, domains):
        '''computes the interval af the taylormodel given the domain.'''
        monointervals = [x.get_interval(domains) for x in self.expansion]

        lb = sum([y[0] for y in monointervals])
        ub = sum([y[1] for y in monointervals])
        return [lb, ub]

    def __str__(self):
        '''returns a string representation of a taylormodel'''
        if len(self.expansion) == 0:
            return str(self.remainder)
        out = ""
        vn = []
        if self.varnames:
            vn = self.varnames
        else:
            # compute temporary var names.
            # number of vars
            nvars = len(self.expansion[0].degrees)
            for i in range(nvars):
                vn.append("temp_var" + str(i))
        for i in range(len(self.expansion)):
            if i != 0:
                out += " + "
            mono = self.expansion[i]
            out += str(mono.coeff)

            vardeg = []
            for j in range(len(mono.degrees)):
                if mono.degrees[j] > 0:
                    vardeg.append((j, mono.degrees[j]))

            if vardeg: out += " * "

            for vd in vardeg:
                out += vn[vd[0]] + '^' + str(vd[1])
        out += " + " + str(self.remainder)
        return out

if __name__ == '__main__':
    examplefile = open('exampleTM.json', 'r')

    jsonex = json.load(examplefile)

    tm = TaylorModel()
    tm.from_json(jsonex["clockG"])

    print(str(tm))
    examplefile.close()
