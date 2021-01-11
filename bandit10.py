import random
import numpy as np
import matplotlib.pyplot as plt
qstar=np.random.randn(1,10)
def bandit(a):
    """
    赌博机反馈价值函数
    """
    t=qstar[0][a]
    return np.random.normal(t,1)
Q=[]
N=[]
for i in range(10):
    Q.append(0)
    N.append(0)
epsi=0.1
for i in range(2000):
    if random.uniform(0.0,1.0)<epsi:
        a=random.randint(0,9)
    else:
        t=max(Q)
        a=Q.index(t)
    r=bandit(a)
    N[a]+=1
    Q[a]+=(1/N[a])*(r-Q[a])

plt.plot(Q)
plt.show()