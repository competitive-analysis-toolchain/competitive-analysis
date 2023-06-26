#
#     This file is part of rockit.
#
#     rockit -- Rapid Optimal Control Kit
#     Copyright (C) 2019 MECO, KU Leuven. All rights reserved.
#
#     Rockit is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     Rockit is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

"""
Car accelerating on a linear track
====================================


"""
from rockit import *
from numpy import sin, pi, linspace
import matplotlib.pyplot as plt
from pylab import *
import csv
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
start_positions = linspace(0,200, num=200)


ocp = Ocp(T=2.0)

# Define constants
m = 500.0
c = 2
d = 1000
F_max = 2500

# Define states
p = ocp.state()
v = ocp.state()

# Defince controls
F = ocp.control()

# Specify ODE
ocp.set_der(p, v)
ocp.set_der(v, 1/m * (F - c * v**2))

# Lagrange objective
ocp.add_objective(ocp.T)

# Path constraints
ocp.subject_to(-F_max <= (F<= F_max))
ocp.subject_to(v >= 0)

# Initial constraints
ocp.subject_to(ocp.at_t0(p)==991.727)
ocp.subject_to(ocp.at_t0(v)==8.36576)

# End constraints
ocp.subject_to(ocp.at_tf(p)==d)
ocp.subject_to(ocp.at_tf(v)==0)

# Pick a solver
ocp.solver('ipopt')

# Choose a solution method
ocp.method(MultipleShooting(N=2,M=1,intg='rk'))

# solve
blockPrint()
sol = ocp.solve()
enablePrint()

# Post-processing
tsa, pa = sol.sample(p, grid='control')
tsa, va = sol.sample(v, grid='control')

# # tsb, pb = sol.sample(p, grid='integrator')
# # tsb, vb = sol.sample(v, grid='integrator')

# # figure(figsize=(10, 4))
# # subplot(1, 2, 1)
# # plot(tsb, pb, '.-')
# # plot(tsa, pa, 'o')
# # xlabel("Times [s]", fontsize=14)
# # grid(True)
# # title('State p')

# # subplot(1, 2, 2)
# # plot(tsb, vb, '.-')
# # plot(tsa, va, 'o')
# # plot(tsa, 1-sin(2*pi*pa)/2, 'r--')
# # legend(['grid_integrator', 'grid_control'])
# # xlabel("Times [s]", fontsize=14)
# # title('State v')
# # grid(True)

tsol, usol = sol.sample(F, grid='control')

for i in range(len(usol)):
    with open('data_set_car_example_delete.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow([pa[i], va[i], usol[i]])


with open('MPC_position_car_example_delete.csv', "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(pa)

with open('MPC_speed_car_example_delete.csv', "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(va)
# # figure()
# # step(tsol,usol,where='post')
# # title("Control signal")
# # xlabel("Times [s]")
# # grid(True)

# # plt.show(block=True)



