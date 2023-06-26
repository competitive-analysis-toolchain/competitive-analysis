import sys
import pylab as plt
import numpy as np
from casadi import Function
import csv
import os
sys.path.insert(0, 'C:\\Users\\Stijn\\Documents\\dirac\\dirac\\SCARA\\wp5\\impact_workflow\\libraries\\scara_ipopt_cartesian_doubleinteg_build_dir')
from impact import Impact

impact = Impact("scara_ipopt_cartesian_doubleinteg",src_dir="C:\\Users\\Stijn\\Documents\\dirac\\dirac\\SCARA\\wp5\\impact_workflow\\libraries\\")

f = Function.load("C:\\Users\\Stijn\\Documents\\dirac\\dirac\\SCARA\\wp5\\impact_workflow\\libraries\\scara_ipopt_cartesian_doubleinteg_build_dir\\integrate_scara_ipopt_cartesian_doubleinteg.casadi")
# Example: how to set a parameter
val_p_end = [0.299999999999999989,0.225000000000000006]
impact.set("p", "p_end", impact.EVERYWHERE, impact.FULL, val_p_end)
        
print("Solve a single OCP (default parameters)")
impact.solve()


# Get scaraParallel.xee solution trajectory
print(impact.get("x_opt", "scaraParallel.xee", impact.EVERYWHERE, impact.FULL))

# Get solution trajectory
x_opt = impact.get("x_opt", impact.ALL, impact.EVERYWHERE, impact.FULL)

# Plotting


_, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(x_opt.T)
ax[0].set_title('Single OCP')
ax[0].set_xlabel('Sample')

print("Running MPC simulation loop")

for x_start in np.linspace(0.09,0.11,5):
  for y_start in np.linspace(0.39,0.41,5):
    history = []
    history_u = []
    x_array = []
    y_array = []
    vx_array = []
    vy_array = []
    
    x_sim = [[x_start], [y_start], [0], [0]]
    impact.set("x_current", impact.ALL, 0, impact.FULL, x_sim)
    
    for i in range(100):
      impact.solve()

      # Optimal input at k=0
      u = impact.get("u_opt", impact.ALL, 0, impact.FULL)
      history.append(x_sim)
      history_u.append(u)

      x_array.append(x_sim[0][0])
      y_array.append(x_sim[1][0])
      vx_array.append(x_sim[2][0])
      vy_array.append(x_sim[3][0])
      # Simulate 1 step forward in time
      # (TODO: use simulation model other than MPC model)
      x_sim = impact.get("x_opt", impact.ALL, 1, impact.FULL)
      
      # Update current state
      impact.set("x_current", impact.ALL, 0, impact.FULL, x_sim)
      

    with open('data_set_scara.csv', "a+") as output:
      writer = csv.writer(output, lineterminator='\n')
      for idx in range(len(history)):
        writer.writerow([history[idx][0][0], history[idx][1][0],  history[idx][2][0],  history[idx][3][0],  history_u[idx][0][0], history_u[idx][1][0]])
    
    with open('MPC_x_scara.csv', "a+") as output:
      writer = csv.writer(output, lineterminator='\n')
      writer.writerow(x_array)

    with open('MPC_y_scara.csv', "a+") as output:
      writer = csv.writer(output, lineterminator='\n')
      writer.writerow(y_array)

    with open('MPC_vx_scara.csv', "a+") as output:
      writer = csv.writer(output, lineterminator='\n')
      writer.writerow(vx_array)

    with open('MPC_vy_scara.csv', "a+") as output:
      writer = csv.writer(output, lineterminator='\n')
      writer.writerow(vy_array)
        
        


# More plotting
ax[1].plot(np.hstack(history).T)
ax[1].set_title('Simulated MPC')
ax[1].set_xlabel('Sample')
plt.show()



      