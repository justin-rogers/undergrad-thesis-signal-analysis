# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:54:45 2018

@author: Justin
"""
import numpy as np
from numpy import inf
from operator import itemgetter
import functools
from math import log2 as lg
from matplotlib import style
style.use('fivethirtyeight')

@functools.lru_cache(maxsize=128) #memoization decorator
def midi_to_hz(n): #converts midi numbers 0-127 to hz, assuming 69=A4=440hz
    dist=n-69
    return 440*2**(dist/12)

def hz_to_midi(n, rnd=True): #inverts above process: returns a midi note number
    if n==0:
        return 0
    try:
        ans=12*(lg(n)-lg(440))+69
        if rnd: #optionally gives an integer value
            ans=round(ans)
        return ans
    except ValueError:
        assert 0, 'valueerror from hz_to_midi on input n: {}'.format(n)

def get_err_new_midi(midi_note, peaks, scale=True):
    """
    To do: Write a different error-getting function.
    Ideally, this will solve the "inverse problem" of overfitting.
    """
    return

def get_err(midi_note, peaks, scale=True):
    G=midi_to_hz(midi_note)
    error=0
    big_amp=max([x[1] for x in peaks])
    for freq,amp in peaks:
        c=int(round(freq/G))
        """
        c is the closest integral multiple of our candidate frequency (freq).
        We check how well it fits the peaks we observe, using L1 norms.
        """
        prediction_error=abs(G*c-freq)
        if scale:
            prediction_error*=(amp/big_amp)
        error+=prediction_error
    """
    To do: Scale this by total persistence, to make the problem
    more amenable to regularization techniques.
    """
    return error/big_amp

def best_guess(peaks, candidates,):
    best_pair=[-1,inf]
    for n in candidates:
        err=get_err(n,peaks)/midi_to_hz(n)
        if err<best_pair[1]:
            best_pair=[n,err]
    return best_pair[0]

def midify_diagram(F, output='hz'):
    #Given a diagram, discretize its frequencies.
    new_dgm=[]
    for freq,amp in F:
        note=hz_to_midi(freq)
        new_dgm.append([note,amp])
        
    F=new_dgm
    for i in range(len(F)):
        try:
            while F[i][0]==F[i+1][0]:
                max_amp=max(F[i][1],F[i+1][1])
                F[i]=[F[i][0],max_amp]
                del F[i+1]
        except IndexError:
            pass
    #Remove duplicates by taking the largest amplitude component
    if output=='midi':
        return F
    elif output=='hz':
        return [[midi_to_hz(l[0]),l[1]] for l in F]

#####Spectral harmonic correlation#####

"""
http://bingweb.binghamton.edu/~hhu1/paper/Zahorian2008spectral.pdf

cf. 1: spectral harmonics correlation

If this is good, we can try a discretized persistence version.
"""
np.seterr(all='raise')

def shc(spec, freq, window_lower, window_upper, harmonic_count=3, discard_unison=False):
    ans=0
    pad_size=harmonic_count*freq
    pad_amt=spec.mean()
    spec=np.pad(spec, [(0,pad_size)], mode='constant', constant_values=(pad_amt))
    for fprime in range(freq-window_lower,freq+window_upper+1): #outer sum
        stop=min(len(spec),(harmonic_count)*freq+fprime)
        #Difference from the Zahorian paper:
        #We take the product from r=0 to h, not r=1 to h+1.
        #Works better and makes sense.
        component_indices=np.arange(fprime,stop,freq)
        try:
            ans+=spec[component_indices].prod()
        except FloatingPointError:
            return np.inf
    return ans

def shc_dft_predict(spec, M, w=None, h=None, first_pass=True):
    """
    Arguments: Spectrum (a fft), M (list of midi candidates),
    w=[wl,wu] where wl,wu are lower/upper bounds for window, h (harmonics to calculate)
    The last two arguments will be filled by automatic values if not given.
    """
    spec=spec.astype('float64')
    if h==None:
        h=int(midi_to_hz(M[-1])/midi_to_hz(M[0]))
        """
        Default behavior: The range of significant frequencies is estimated from
        the midi candidates, and a number of harmonics is selected to span the range.
        """
    candidates=[]
    for m in M:
        freq=int(4*midi_to_hz(m)) #units of 0.25 hz
        if w==None:
            #By default: Takes the preimage of hz_to_midi(m).
            #Get absolute bounds:
            abs_l=int(4*midi_to_hz(m-0.5)+1) #round up
            abs_u=int(4*midi_to_hz(m+0.5)) #round down
            #given in units of 0.25hz, to be compatible with FFT resolution
            #Get relative bounds to the frequency in question:            
            wl=freq-abs_l
            wu=abs_u-freq
        elif len(w)==2:
            wl, wu=w
        m_val=shc(spec,freq,wl,wu,h, discard_unison=False)
        candidates.append([m_val,m])
        
    candidates=sorted(candidates) #sorts by m_val ascending
    if candidates[-1][0]==np.inf: #overflow problems
        scaled_spec=spec/536870912.0  #536870912.0==np.finfo('float32').eps/np.finfo('float64').eps
        return shc_dft_predict(scaled_spec, M)
    
    ans=candidates[-1][1] #m with biggest val
    if first_pass==True:
        #We sweep through again looking to avoid octave errors.
        ans=midi_to_hz(ans)
        new_candidates=[hz_to_midi(x) for x in [ans/3, ans/2, ans, ans*2, ans*3] if 21<=hz_to_midi(x)<=108]
        return shc_dft_predict(spec, new_candidates, h=h, first_pass=False)
    return ans

#####Persistence#####

def persist0(data, significance_threshhold=.05, start_index=1, high_pass=False):
    """
    Input: Data (expecting a fft).
    
    Output: The 0th persistence diagram of that data, in the following form:
    [[peak1, persistence1], [peak2, persistence2], ...]. peak1 is units of hz.
    peak2 is persistence in the codomain, i.e., the difference in frequency
    between the time of birth and the time of death, for a given component.
    
    Parameters:
        0<significance_threshhold<1 will throw away all components
    with persistence smaller than max(data)*significance_threshhold.
        
        start_index allows you to throw away the first n elements of your data.
    This is useful because the 0th element of the fft array contains no pitch info.
    
        high_pass changes start_index to the value of 107, throwing away all low
    pitch information for the diagram. This discards information about pitches
    that are too low to be in the dataset, and avoids a couple of errors.
    
    Algorithm:
    This is an original "simplification"/improvement of the more general union-find
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
    #Default is to remove the 0th term, which contains no pitch information.
    #Another desirable value is 107, which starts at the lowest frequency
    #that we care about predicting (midi 21).
    if high_pass:
        start_index=107
    data=data[start_index:] #Slice off the beginning of data.
    
    significance_cap=max(data)*significance_threshhold
    #We drop all components with significance below this-- default 5% of max.
    
    node_count=len(data)
    nodes=sorted(list(range(node_count)), key=lambda n: data[n])
    
    lpts=[]
    rpts=[]
    births=[]
    xfreqs=[]
    #Keep the above lists updated together: for a fixed index i,
    #lpts[i], rpts[i], births[i] should give the component's info.
    dead_components=[]
    for t,x in enumerate(nodes[::-1]):
        if data[x]<significance_cap:
            break
        
        xr=x in lpts #boolean: x is a right attaching point
        if xr:
            b=lpts.index(x) #b is the component that is right of x
        xl=x in rpts
        if xl:
            a=rpts.index(x) #a is the component left of x
        
        if xr and xl: #if it attaches to 2 components, younger dies
            if births[a]<births[b]: #if a is the older component (born first)
                #then a eats b, and steals its right attpt
                rpts[a]=rpts[b]
                
                persistence=data[xfreqs[b]]-data[x] #amplitude of birth - amplitude of death
                #I suspect this may cause problems in certain situations: e.g., 3 consecutive freqs
                #with values (10000, 5, 20000)
                if persistence>significance_cap:
                    dead_components.append([(xfreqs[b]+start_index)/4, persistence]) #freq, persistence
                del lpts[b]
                del rpts[b]
                del births[b]
                del xfreqs[b]
            else: #b eats a
                lpts[b]=lpts[a]
                persistence=data[xfreqs[a]]-data[x]
                if persistence>significance_cap:
                    dead_components.append([(xfreqs[a]+start_index)/4, persistence])
                del lpts[a]
                del rpts[a]
                del births[a]
                del xfreqs[a]
        elif xl: #if x is an attaching point for a left component, increase its right attpt by 1
            rpts[a]+=1
        elif xr: #decrease left attpt by 1
            lpts[b]-=1
        else: #x is not an attaching point anywhere, so it creates a new component
            lpts.append(x-1)
            rpts.append(x+1)
            births.append(t)
            xfreqs.append(x)
    for i in range(len(xfreqs)): #When we exit the loop, change live components to dead ones.
        dead_components.append([(xfreqs[i]+start_index)/4, data[xfreqs[i]]])
    F=sorted(dead_components, key=itemgetter(0))
    F=[[a,b-significance_cap] for a,b in F]
    """
    Above subtracts the persistence truncation. We do this so if a component is
    born at (cap+1) and dies at (cap-1), we don't call it significant-- it has 
    1 persistence. This might cut short some missing fundamentals, but we can
    infer them anyway-- and we can tune the significance cap parameter.
    
    This minimizes the number of undesirable high-persistence artifacts of the
    significance-cap truncation.
    """
    return [F, calc_midi(F)]

def calc_midi(F):
    #Given a diagram, convert each frequency to a midi note, and return those notes.
    return sorted(list(set([hz_to_midi(l[0]) for l in F if 21<=hz_to_midi(l[0])])))