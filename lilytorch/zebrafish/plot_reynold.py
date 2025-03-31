
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rc('font', **{"size":25})
plt.rcParams["figure.figsize"] = (6,6)

names = [
    "Tadpole (larval)",
    "Tadpole (adult)",
    "Zebrafish (adult)",
    "Salamander (adult)"
]
lengths = np.array([
    0.005,
    0.025,
    0.044,
    0.1632
])
speeds = np.array([
    0.05,
    0.13,
    0.05,
    0.1
])
Re = np.array([
    250,
    3250,
    7500,
    16320,
])

markers = [
    "d",
    "v",
    "o",
    "s"
]

Re = (10**6)*(speeds*lengths)
print(Re)

for i in range(4):
    plt.loglog(lengths[i], Re[i], marker=markers[i], markersize=20, label=names[i])
    plt.plot([0,0.2],[1000,1000])
    plt.fill_between([0,0.2],[0,0],[1000,1000], alpha=0.05)
    plt.fill_between([0,0.2],[1000,1000], [50000,50000], alpha=0.05)
plt.xlim([0,0.2])
plt.ylim([0,50000])
plt.xlabel("Length (m)")
plt.ylabel("Re")
plt.legend(loc="upper left")

plt.savefig("intertial_regime.pdf")