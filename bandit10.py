import random
import numpy as np
import matplotlib.pyplot as plt
qstar=[0.3,-0.6,1.5,0.6,1.3,-1.4,-0.2,-1,0.8,-0.3]#np.random.randn(1,10)
def bandit(a):
    """
    赌博机反馈价值函数
    """
    t=qstar[a]
    r=np.random.normal(t,1)
    return r
Q=[]
N=[]
for i in range(10):
    Q.append(0)
    N.append(0)
epsi=0.1
rl=[]
rmax=-np.inf
k=0
for i in range(200):
    if random.uniform(0.0,1.0)<epsi:
        a=random.randint(0,9)
        k+=1
    else:
        g=max(Q)
        a=Q.index(g)
    r=bandit(a)
    # if r>rmax:
    #     rmax=r
    rl.append(r)
    N[a]+=1
    Q[a]+=(1/N[a])*(r-Q[a])

plt.plot(rl)
print(k)
plt.show()