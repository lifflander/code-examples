from parser import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

d = dict()

iters = [10, 20, 30, 40, 50, 100, 200, 500]

for i in iters:
    d["nograph"] = dict()
    d["graph"] = dict()

for i in iters:
    d["graph"][i] = readTranspose(
        "./graph.1.dat",
        [0, 1, 2, 3, 4, 5],
        ["N", "iters", "ftime", "fstd", "ptime", "pstd"],
        [1, i]
    )

for i in iters:
    d["nograph"][i] = readTranspose(
        "./graph.0.dat",
        [0, 1, 2, 3, 4, 5],
        ["N", "iters", "ftime", "fstd", "ptime", "pstd"],
        [1, i]
    )

################################################################################

N = d["graph"][iters[0]]["N"]
# N = N[0:1]

print(N)

s = dict()

for num in N:
    s[num] = dict()

for i in iters:
    for n, num in enumerate(N):
        gtime = d["graph"][i]["ftime"][n]
        ngtime = d["nograph"][i]["ftime"][n]
        speedup = ngtime / gtime
        s[num][i] = speedup
        # print("n={}, i={}: {} -- s={}".format(n, i, gtime, speedup))


matplotlib.rcParams.update({'font.size': 18})

c = cm.get_cmap("Set1")

fig, ax = plt.subplots(figsize=(getLongGraphLen(),getLongGraphWidth()))

# print(d["nograph"][10])
# print(d["graph"][10])

cl = 0.12
width = 0.1

x = []
x.append(np.arange(len(N)))
for n, _ in enumerate(N):
    x.append([i + width for i in x[n]])

y = []
for _, it in enumerate(iters):
    lst = []
    for n in N:
        lst.append(s[n][it])
    y.append(lst)

print(len(x))
print(len(y))
print(x[0])
print(y[0])

for i, xi in enumerate(y):
    ax.bar(x[i], y[i], width, color=c(i*cl), label="iter="+str(iters[i]), zorder=3)

ax.legend(loc='upper right', ncol=2, fontsize=14)
ax.yaxis.labelpad=16
ax.set_ylabel('Speedup')

plt.xlabel('$N$', fontweight='bold')
plt.xticks([r + width for r in range(len(N))], [int(N[k]) for k in range(len(N))], fontsize=7)

plt.ylim([0, 1.8])
plt.rc('font', size=18)

plt.title("Sample Program A (Speedup)")

addAllGridLines(ax)

writeFile(__file__, fig)
