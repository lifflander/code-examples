
def read(f, where=[]):
    f = open(f)
    lines = f.readlines()

    matrix = []
    for l in lines:
        l = l.rstrip()
        arr = l.split(' ')
        #print arr
        if len(where) == 0:
            matrix.append(arr)
        else:
            if int(arr[where[0]]) == where[1]:
                matrix.append(arr)
    return matrix

def transpose(data, select, names=[]):
    out = []
    for x in range(len(select)):
        out.append([])
    for i, elm in enumerate(select):
      for j in data:
          out[i].append(float(j[elm]))
    if len(names) == 0:
        return out
    else:
        r = dict()
        for i, n in enumerate(names):
            r[n] = out[i]
        return r

def readTranspose(f, selector, names=[], where=[]):
    matrix = read(f, where)
    return transpose(matrix, selector, names)


def writeFile(f, fig):
    name = f[0:len(f)-2] + "pdf"
    fig.savefig(name, dpi=600)


def addAllGridLines(ax):
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

def getLongGraphLen():
    return 10

def getLongGraphWidth():
    return 6
