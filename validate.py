import sys
import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

num_samples = 10000
dataID = sys.argv[1]
dataPath = 'data/'+dataID+'/'
resultPath = 'results/'+dataID+'/'

# Pickle Load
def pload( path ):
    dump = pickle.load( open( path, 'rb' ) )
    return( dump )

def preprocess(X):
    dim = len(X[0])
    return( np.reshape(np.transpose(X,(0,2,1,3)),(-1,dim,20)) )

def get_du(h,v):
    h = np.square(h)
    v = np.square(v)
    return(np.matmul(np.transpose(h,(1,0)),v))

def get_dv(h,u,w):
    h = np.square(h)
    u = np.square(u)
    u = np.multiply(u,w)
    return(np.matmul(h,u))

H = pload(dataPath+'test_H.pkl')
V = preprocess(pload(resultPath+'vlist.pkl'))
Ve = preprocess(pload(resultPath+'vlist_e.pkl'))
U = preprocess(pload(resultPath+'ulist.pkl'))
Ue = preprocess(pload(resultPath+'ulist_e.pkl'))
W = preprocess(pload(resultPath+'wlist.pkl'))
We = preprocess(pload(resultPath+'wlist_e.pkl'))
A = preprocess(pload(resultPath+'alist.pkl'))
Ae = preprocess(pload(resultPath+'alist_e.pkl'))


# RHS
bound = np.zeros((num_samples,5,20))
# LHS
diff = np.zeros((num_samples,5,20))

for i in range(num_samples):
    hh = H[i]
    for j in range(1,5):
        Du = get_du(hh,V[i,j,:])
        for k in range(20):
            diff[i,j,k] = np.absolute(V[i,j,k] - Ve[i,j,k])
            e1 = (hh[k,k]**2/Du[k]) * ( np.absolute(V[i,j-1,k] - Ve[i,j-1,k]) + ((hh[k,k]**2/Du[k]) * np.absolute(A[i,j-1,k]*V[i,j-1,k] - Ae[i,j-1,k]*Ve[i,j-1,k]) ) )
            e2 = 0.0
            for kk in range(20):
                e2 += (hh[kk,kk]**2/Du[kk]) * ( np.absolute(V[i,j-1,kk] - Ve[i,j-1,kk]) + ((hh[kk,kk]**2/Du[kk]) * np.absolute(A[i,j-1,kk]*V[i,j-1,kk] - Ae[i,j-1,kk]*Ve[i,j-1,kk])))
            bound[i,j,k] = (e1 + Ve[i,j,k]*e2)/get_dv(hh[k,:],U[i,j-1,:],W[i,j-1,:])

# Plots        

# 3D plot of LHS and RHS
x = range(4)
y = range(20)
indx = np.random.randint(10000)
b = bound[indx][1:,:]
d = diff[indx][1:,:]
X,Y = np.meshgrid(x,y)
fig = plt.figure(figsize=(12,8),tight_layout=True)
ax = plt.axes(projection='3d')
ax.scatter(X, Y, b, zdir='z', c= 'red',label='Upper bound')
ax.scatter(X, Y, d, zdir='z', c= 'blue',label='Observed error')
ax.set_xlabel('Layers',fontsize=12)
ax.set_ylabel('Nodes',fontsize=12)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(fontsize=12)
plt.savefig('Figure_1.png')

# Histogram of difference
boundh = np.reshape(bound,(-1,))
diffh = np.reshape(diff,(-1,))
violations = boundh - diffh
violations[np.where(violations>5.0)] = 5.0 
fig, axs = plt.subplots(1, 1, tight_layout=True)
N, bins, patches = axs.hist(violations, bins=1000,density=True)
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.xlim(-0.5,1.0)
plt.ylim(0.0,5.0)
axs.set_ylabel('Percentage of samples',fontsize=12)
axs.set_xlabel('RHS - LHS',fontsize=12)
axs.yaxis.set_major_formatter(PercentFormatter())
plt.savefig('Figure_2.png')
