import time
from csp import *
import pandas as pd
import sys


class exam_schedule(CSP):

    def __init__(self, csv):
        df = pd.read_csv(csv)
        self.variables = []
        self.domains = {}
        self.neighbors = {}
        self.weights = {}
     
        self.spots = []
        self.kathigitis = {}
        self.eksamino = {}
        self.has_lab = {}
        self.is_hard = {}


        for i in range(21):
            for j in range(3):
                self.spots.append((i+1,j+1))

        print(len(self.spots))
        for index,row in df.iterrows():
            self.variables.append(row['Μάθημα'])
            self.kathigitis[row['Μάθημα']] = row['Καθηγητής']
            self.eksamino[row['Μάθημα']] = row['Εξάμηνο']
            self.has_lab[row['Μάθημα']] = row['Εργαστήριο (TRUE/FALSE)']
            self.is_hard[row['Μάθημα']] = row['Δύσκολο (TRUE/FALSE)']
        
        for lesson in self.variables:
            if self.has_lab[lesson]:
                erg = lesson + " ERGASTHRIO"
                self.variables.append(erg)
                self.kathigitis[erg] = self.kathigitis[lesson]
                self.eksamino[erg] = self.eksamino[lesson]
                self.has_lab[erg] = False
                self.is_hard[erg] = False

        print(len(self.variables))
        for lesson in self.variables:   # add domains
            self.domains[lesson] = self.spots
            

        for lesson in self.variables:
            self.neighbors[lesson] = []
            for neighbor in self.variables:
                if neighbor == lesson:
                    continue
                self.neighbors[lesson].append(neighbor)

        for a in self.variables:
            for b in self.neighbors[a]:
                self.weights[(a,b)] = 1

        CSP.__init__(self, self.variables, self.domains, self.neighbors, self.var_constraints)
    
    

    def var_constraints(self, A, a, B, b):
        
        if a == b:

            return False
        
        if self.eksamino[A] == self.eksamino[B] or self.kathigitis[A] == self.kathigitis[B]:
            if A == B + " ERGASTHRIO":
                if a[0] != b[0] or a[1] != b[1] + 1:
                    return  False
            elif B == A + " ERGASTHRIO":
                if a[0] != b[0] or b[1] != a[1] + 1:
                    return False
            else:
                if a[0] == b[0]:
                    return False
        if self.is_hard[A] and self.is_hard[B]:
            if abs(a[0]-b[0]) <= 1:
                return False

        return True
    
    def incr_weight(self,A,B):
        self.weights[(A,B)] += 1

    
    
    def display(self,assignment):
        lessons = list(assignment.keys())
        values = list(assignment.values())
        for spot in self.spots:
            if spot in values:
                if spot[1] == 1:
                    position = values.index(spot)
                    print(f"day {spot[0]} hour 9:00-12:00 : {lessons[position]} eksamino {self.eksamino[lessons[position]]} teacher {self.kathigitis[lessons[position]]}")
                if spot[1] == 2:
                    position = values.index(spot)
                    print(f"day {spot[0]} hour 12:00-15:00 : {lessons[position]} eksamino {self.eksamino[lessons[position]]}  teacher {self.kathigitis[lessons[position]]}")
                if spot[1] == 3:
                    position = values.index(spot)
                    print(f"day {spot[0]} hour 15:00-18:00 : {lessons[position]} eksamino {self.eksamino[lessons[position]]}  teacher {self.kathigitis[lessons[position]]}")

        for val in self.weights.values():
            if val > 1:
                print(val)
        print(f"nodes :{self.nassigns}")

    def write_to_csv(self,csv,assignment):
        
        lessons = list(assignment.keys())
        values = list(assignment.values())
        myarray = []
        for i in range(21):
            myarray.append(["","",""])
        for i in range(21):
            for j in range(3):
                spot = (i+1,j+1)
                if spot in values:
                    position = values.index(spot)
                    myarray[i][j] += (lessons[position] + " (" + str(self.eksamino[lessons[position]]) + ")")

        
        

        df = pd.DataFrame(myarray,columns=["9:00-12:00","12:00-15:00","15:00-18:00"])
        df.to_csv(csv,index=False)

########### BACKTRACKING ALGORITHMS ###########################################



# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return csp.first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def dom_wdeg(assignment,csp):
        
        min_var = (None,float("inf"))
        for A in csp.variables:
            if A in assignment:
                continue
            dom = len(csp.choices(A))
            wdeg = 1
            for B in csp.neighbors[A]:
                if B in assignment:
                    continue
                wdeg += csp.weights[(A,B)]
            if min_var[0] == None or dom/wdeg < min_var[1]:
                min_var = (A,dom/wdeg)

                
        return min_var[0]

def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])


# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                csp.incr_weight(var,B)                                      # incrementing here is the wipeout
                csp.incr_weight(B,var)
                return False
    return True


# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                csp.incr_weight(Xi,Xj)                      # incrementing here is the wipeout
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)



# The search, proper


def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result


# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs


def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


# ______________________________________________________________________________

if __name__ == "__main__" :



    z = exam_schedule("stoixeiaMathimatwn.csv")

    commands =  { "fc": forward_checking,
                  "mc": min_conflicts, 
                  "mac": mac,
                  "mrv": mrv,
                  "dw": dom_wdeg  
                    }   

    #assert sys.argv[1] not in commands or sys.argv[2] not in commands
    start = time.time()
    if sys.argv[1] == "mc":
        z.display(min_conflicts(z))
        z.write_to_csv("schedulemc.csv",min_conflicts(z))
    else:
        backtracking_search(z,commands[sys.argv[2]],lcv,commands[sys.argv[1]])
        z.display(z.infer_assignment())
        z.write_to_csv("schedule.csv",z.infer_assignment())
    
    end = time.time() - start

    print(f"time taken is : {end}")





