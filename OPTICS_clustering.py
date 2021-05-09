import os,glob,math,sys
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter, defaultdict
import numpy as np

def fetchscore(filename):
    with open(filename, "r") as Structure:
        for line in Structure:
            if line.startswith('Affinity'):
                line = line.split()
                return float(line[1])

class BoxSize():
    def __init__(self, xcm, ycm, zcm, rg):
        self.xcm = xcm
        self.ycm = ycm
        self.zcm = zcm
        self.rg = rg

def lig_size(filename):

    coord_x = []
    coord_y = []
    coord_z = []
    with open(filename, "r") as Structure:
        for line in Structure:
            try:
                if line[0:6] == "HETATM" or line[0:4] == "ATOM":
                    coord_x.append(float(line[30:38]))
                    coord_y.append(float(line[38:46]))
                    coord_z.append(float(line[46:54]))
            except:
                print ("Unknown error detected.")

        N = len(coord_x)

        xcm = sum(coord_x)/N
        ycm = sum(coord_y)/N
        zcm = sum(coord_z)/N

        xg = [i - xcm for i in coord_x]
        yg = [i - ycm for i in coord_y]
        zg = [i - zcm for i in coord_z]

        rg2 = 0.0

        for i in range(N):
            rg2 += (xg[i]**2 + yg[i]**2 + zg[i]**2)/N

        rg = math.sqrt(rg2)

        return BoxSize(xcm, ycm, zcm, rg)


name = "" #Choose file name
srcdir   = "/results/Blinddocking/" + name
ligpdbqt = "/data/Ligands/pdbqt/" + name + ".pdbqt"
ligsize = lig_size(ligpdbqt)
ligcm = np.array([ligsize.xcm,ligsize.ycm,ligsize.zcm])
score_only = "/results/Redocking/" + name + "_score_only.log"
ligaffinity = fetchscore(score_only)

os.chdir(srcdir)


f = np.loadtxt("centerofmass.txt")
Y = np.delete(f, [3,4], axis=1)
energies = np.delete(f,[0,1,2,4], axis=1)
rgyr = np.delete(f,[0,1,2,3], axis=1)
X=np.vstack([Y,ligcm])
e=np.vstack([energies,ligaffinity])
rg=np.vstack([rgyr,ligsize.rg])
length=len(X)

print("Total number of points to cluster: ",length)
clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05) 


# Performing fit
clust.fit_predict(X)
pred=clust.fit_predict(X)
numclust=format(len(set(clust.labels_))-1)

print("Number of clusters found: ",numclust)
print("Cluster populations:",Counter(clust.labels_))

cluster_start_end = []
for i in range(0,int(numclust)):
    for j in range(len(clust.cluster_hierarchy_)):
        hierrange=int(clust.cluster_hierarchy_[j][1]-clust.cluster_hierarchy_[j][0])+1
        if hierrange == int(Counter(clust.labels_)[i]):
            cluster_start_end.append([clust.cluster_hierarchy_[j][0],clust.cluster_hierarchy_[j][1]+1])
print(cluster_start_end)

cluster_xcm = []
cluster_ycm = []
cluster_zcm = []
cluster_averE = []
cluster_averRg = []
cluster_pop = []

pointsoutside=length

for i in range(int(numclust)):
    pop=0
    xcm = 0.0
    ycm = 0.0
    zcm = 0.0
    averE = 0.0
    averRg = 0.0
    print("For cluster ",i)
    for j in range(int(cluster_start_end[i][0]),int(cluster_start_end[i][1])): 
        pop+=1
        index=clust.ordering_[j]
        xcm += X[index][0]
        ycm += X[index][1]
        zcm += X[index][2]
        averE += e[index]
        averRg += rg[index]
    pointsoutside -= pop
    cluster_xcm.append(xcm/pop)
    cluster_ycm.append(ycm/pop)
    cluster_zcm.append(zcm/pop)
    cluster_averE.append(averE/pop)
    cluster_averRg.append(averRg/pop)
    cluster_pop.append(pop)
    print("center of mass : %.2f %.2f %.2f" % (cluster_xcm[i],cluster_ycm[i],cluster_zcm[i]))
    print("average energy : %.2f" % cluster_averE[i])
    print("average radius of gyration : %.2f\n" % cluster_averRg[i])
    print("population: %d\n" % pop)

# Plot data
fig=plt.figure()
fig.suptitle(name)
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(X[:,0], X[:,1], X[:,2], c=clust.labels_, s=300)
colors=['b','g','r','c','m','y','k']
print("blue: 0, green: 1, red: 2, cyan: 3, magenta: 4, yellow: 5, black: 6")
for i in range(int(numclust)):
    ax.plot([cluster_xcm[i]],[cluster_ycm[i]],[cluster_zcm[i]],c=colors[i], marker='x', markersize=75, alpha=1.0)

plt.savefig(output + '.png'
plt.show()
