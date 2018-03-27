# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 11:27:30 2018

@author: Justin
"""
from time import time
from wavedata import WaveData
import matplotlib.pyplot as plt

def persist0(data):
    """
    This is a simplification/improvement of the more general union-find
    approach to computing the 0th persistent homology. It assumes our graph looks
    like this: o-o-...-o-o. Nodes are integers between 0 and N, and two nodes i,j
    are connected if |i-j|=1.
    
    We take advantage of two facts:
        1. The persistence diagram of a graph may be reduced to the following:
        Given connected nodes u,w and some simplex K_j in our filtration,
        determine if u and w belong to the same component. This is computable in
        approximately linear time through union-find data structures.
        
        2. Our particular graph is extremely simple. Instead of storing all the
        nodes of each component, we may store the endpoints-- since a component
        always looks like {j, j+1, j+2, ..., k}, just store [j,k].
        Better yet, we may store the points *adjacent* to the endpoints-- what
        might be called "attaching points". So we store [j-1, k+1], then check
        if a new node might attach.
    
    The basic structure of my algorithm is this:
        1. We have, say, 64000 nodes-- indices of our data points. Sort the indices
        descending by their values in the dataset. (so, e.g., [500,12,17]-->[0,2,1])
        
        2. Keep track of live/dead components as lists:
        [left_attpt, right_attpt, birthtime, deathtime].
        
        3. Iterate: pop a node x from our sorted list.
        (This is the "next" node with respect to sublevel sets.)
        Either x will create a new component, extend an existing component, or
        join together two existing components.
        This corresponds to x being an attaching point for 0, 1, or 2 existing components.
        
        If a component is killed, move it to the list of dead components.
        
        4. At the end of the procedure, move all remaining live components to
        the list of dead components, with a death value of -1 to indicate infty.
        
    """
    
    node_count=len(data)
    nodes=sorted(list(range(node_count)), key=lambda n: data[n])
    #sorts ascending so .pop gives us smallest element
    
    lpts=[]
    rpts=[]
    births=[]
    xbirths=[]
    #Keep the above lists updated together: for a fixed index i,
    #lpts[i], rpts[i], births[i] should give the component's info.
    
    dead_components=[]
    for t in range(node_count):
        x=nodes.pop() #next node
        L_idx=[]
        R_idx=[]
        
        try: #attempt to extract indices where x is a left/right attpt
            R_idx.append(lpts.index(x))
        except ValueError:
            pass
        try:
            L_idx.append(rpts.index(x))
        except ValueError:
            pass

        if L_idx and R_idx: #if both lists are nonempty, i.e., joining 2 components
            a,b=L_idx[0], R_idx[0] #a is the (index of) the component to the left of x, b to the right.
            if births[a]<births[b]: #if a is the older component (born first)
                #then a eats b, and steals its right attpt
                rpts[a]=rpts[b]
                dead_components.append([data[x]-data[xbirths[b]], xbirths[b], births[b], t, lpts[b], rpts[b]]) #entomb R. t=death time
                del lpts[b]
                del rpts[b]
                del births[b]
                del xbirths[b]
            else: #b eats a
                lpts[b]=lpts[a]
                dead_components.append([data[x]-data[xbirths[a]], xbirths[a], births[a], t, lpts[a], rpts[a]])
                del lpts[a]
                del rpts[a]
                del births[a]
                del xbirths[a]
        elif L_idx: #if x is an attaching point for a left component, increase its right attpt by 1
            a=L_idx[0]
            rpts[a]+=1
        elif R_idx: #decrease left attpt by 1
            b=R_idx[0]
            lpts[b]-=1
        else: #x is not an attaching point anywhere, so it creates a new component
            lpts.append(x-1)
            rpts.append(x+1)
            births.append(t)
            xbirths.append(x)
    dead_components.append([-1, births[0], node_count-1, lpts[0], rpts[0]])
    return dead_components
            


q=WaveData()
#start_dat=time()
#D=persist0(q.data)
end_dat=time()
F=persist0(q.fft)
end_fft=time()
#print('data took: {} secs'.format(end_dat-start_dat))
print('fft took: {} secs'.format(end_fft-end_dat))

plt.figure(figsize=(12,12))
plt.scatter([x[1] for x in F][:4000], [-x[0] for x in F][:4000], marker='.')
#plt.scatter([x[1] for x in F], [x[2] for x in F])
#plt.scatter([q.fft[x[1]] for x in F], [q.fft[x[2]] for x in F])
plt.show()
q.graph_fft()

"""
Graph w/ fft[0] included and not.


q.metadata['pitch']
Out[63]: 70

midi_to_hz(70)
Out[64]: 466.1637615180899

466*4
Out[65]: 1864

L=[x for x in F if x[0]>max([x[0] for x in F])/4]

L
Out[67]: 
[[37255277.273865834, 5602, 23639, 31974, 5594, 5608],
 [38744292.066292986, 5570, 25966, 31976, 5566, 5580],
 [42696456.22051046, 5630, 22401, 31979, 5623, 5637],
 [42861762.124941729, 5555, 24080, 31980, 5551, 5565],
 [46365121.466769435, 5588, 22148, 31983, 5565, 5608],
 [61416846.923906356, 3723, 24217, 31986, 3715, 3729],
 [64643481.765285209, 1859, 26194, 31988, 1850, 1864],
 [66990824.062591434, 1872, 24075, 31989, 1865, 1879],
 [68046221.640150547, 5612, 20685, 31990, 5565, 5623],
 [73336549.56168434, 4575, 18, 31991, 3758, 5565],
 [74519498.882181332, 3708, 24853, 31992, 3700, 3714],
 [92922525.021151125, 3750, 21911, 31994, 3744, 3758],
 [125816141.25276132, 3732, 22013, 31997, 3715, 3744],
 [128313153.70098209, 648, 12, 31998, 0, 1865],
 [142750121.59852353, 2953, 5, 32000, -1, 3715]]


Note these last two entries-- freaky!
One idea: track the 0 component. Where does it go?
(And the -1 component-- which is whatever eats the 0 component.)
"""