import matplotlib.pyplot as plt
import numpy as np

actions = []

with open("actions.txt", "r") as f:
    for line in f.readlines():
        line = line[1:-2]           # get rid of brackets
        line = line.split(", ")     # split on commas
        line = list(map(float, line))
        actions.append(line)

actions = np.array(actions)

fig,ax = plt.subplots(3)
ax[0].plot(actions[:,0], color="red")
ax[0].set_title("Turn")
ax[1].plot(actions[:,1], color="green")
ax[1].set_title("Gas")
ax[2].plot(actions[:,2], color="blue")
ax[2].set_title("Brake")
plt.show()