from parser import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

d = dict()

iters = [10, 20, 30, 40, 50, 100, 200, 500]

for i in iters:
    d["graph"] = dict()

for i in iters:
    d["graph"][i] = readTranspose(
        "./graph.1.dat",
        [0, 1, 2, 3, 4, 5],
        ["N", "iters", "ftime", "fstd", "ptime", "pstd"],
        [1, i]
    )

################################################################################

matplotlib.rcParams.update({'font.size': 18})

c = cm.get_cmap("Set1")

fig, ax = plt.subplots(figsize=(getLongGraphLen(),getLongGraphWidth()))

print(d["graph"][10])

x = 0.12
for j, i in enumerate(iters):
    ax.errorbar(d["graph"][i]["N"], d["graph"][i]["ftime"], d["graph"][i]["fstd"], label="Iter=" + str(i), linewidth=1.6, color=c(j*x), marker='^', linestyle='--')

ax.legend(loc='upper left', fontsize=14)
ax.yaxis.labelpad=16
ax.set_xlabel('$N$')
ax.set_ylabel('Per iteration time (μs)')

ax.set_xscale('log')

# ax.set_yscale('log')
# plt.ylim([10, 1000])
plt.ylim([0, 420])

# plt.xlim([0, 1600])
plt.rc('font', size=18)

plt.title("Sample Program A (Graph)")

# plt.text(-150, 1.25, '(d)', fontsize=20.0, va='center')

addAllGridLines(ax)

writeFile(__file__, fig)
