import numpy as np
import math 
from scipy.optimize import newton
from scipy.special import spence,loggamma
from math import log
import numpy.random as nr
import graph_tool as gt
import random as rd
import pickle as pk
import igraph as ig
import scipy.misc as sm
from scipy.special import comb
from graph_tool.topology import subgraph_isomorphism
from graph_tool.all import remove_parallel_edges,remove_self_loops
import graph_tool.collection as gtc

import time
from multiprocessing import Pool, Manager,get_context
#igraph convertion 
def toig(g2):
    return ig.Graph(list(g2.get_edges()[:,[0,1]]),directed=g2.is_directed())
import os
def get_queries(dir): #get list of networks in directory
    queries=[]
    for root, dirs, filenames in os.walk(dir):
        for f in filenames:
            if f[-2:]=="ml": 
                queries.append(os.path.join(root, f))
    queries.sort()
    return queries
#combinatorial funtions
def logfac(n):
    return loggamma(np.array(n)+1)

def logDB(n,m):
    return logBI(n+m-1,m)
def logBI(n,m):
    if m>=n:
        return 0.0
    else:
        return logfac(n)-logfac(m)-logfac(n-m)

def ilog(n):
    v=math.log(2.865,2)
    s=math.log(n,2)
    while s>0:
        v+=s
        s=math.log(s,2)
    return v*math.log(2)
def logStar(n):
    v=math.log(3.4667)
    s=math.log(n)
    while s>0:
        v+=s
        s=math.log(s)
    return v
def logFD(n):
    if n<14:
        return math.log(math.factorial(n))
    else:
        return (n+0.5)*(math.log(n))-n+0.5*math.log(6.28)+1.0/(12*n)
def logBID(n,m):
    if m>=n:
        return 0.0
    elif n<10**10:
        return logFD(n)-logFD(m)-logFD(n-m)
    else:
        return m*log(n)-logfac(m)
from scipy.special import factorial
#Degree sequence priors 
def StoD(s):#convert degree sequence s to degree distribution
    y = np.bincount(s)
    ii = np.nonzero(y)[0]
    return ii,y[ii] 
def stod(s):
    D=[]
    for i in range(len(s)):
        y=np.bincount(s[i])
        ii=np.nonzero(y)[0]
        D.append(y[ii])
    return D
def stodZero(s):
    D=[]
    zeros=[]
    for i in range(len(s)):
        y=np.bincount(s[i])
        ii=np.nonzero(y)[0]
        ii=np.delete(ii,0)
        zeros.append(y[0])
        D.append(y[ii])
    return D,zeros
def entD(d,N):#log P(d_m,o|nk_m,o)
    return sum([logfac(N)-sum([logfac(n) for n in s]) for s in d])

def logq(m,n):
    if m==0:
        return 0
    elif n<m**(1/4.0):
        return logBI(m-1,n-1)-logfac(n)
    else:
        u=n/math.sqrt(m)   
        func= lambda v : u*math.sqrt(spence(math.exp(-v)))-v  
        v=newton(func,u*1.28)
        return log(v)-0.5*log(1-(1+(u**2)/2.0)*math.exp(-v))-log(2)*1.5-log(u)-log(3.14)-log(m)+math.sqrt(m)*(2*v/u-u*log(1-math.exp(-v)))
def slambda(E,em):
    fun=lambda v:E-sum([float(i)/(1-v**i) for i in em])
    return newton(fun,math.exp(-len(em)/float(E)))

def ents(E,em):#-log(p(nm|E,M))
    alpha=slambda(E,em)
    return -sum([math.log((alpha**-t)-1) for t in em])-math.log(alpha)*E
def entapp(E,em):#-log(p(nm|E,M))
    return len(em)*(log(E/float(len(em)))+1)-sum(np.log(em))

def logDH(m,nm,N):#homogeneous prior 
    return sum([logDB(N,len(o)*nm) for o in m.gp.orbits])
def logDHNO(m,nm,N):#homogeneous prior no orbit
    return logDB(N,nm*m.num_vertices())
def logDHA(M,nm,N):#homogeneous prior total atomic degree 
    td=sum([M[i].num_vertices()*nm[i] for i in range(len(M))])
    return logDB(N,td)
def logD1(d,N):#independent prior -log(P(d_m,i|m,nm))
    C=stod(d)
    return sum([min(logDB(N,np.sum(d[i])),entD([C[i]],N)+logq(np.sum(d[i]),N)) for i in range(len(d))])
    #return sum([entD([C[i]],N)+logq(np.sum(d[i]),N) for i in range(len(d))])
def DOtoDA(D):
    return np.sum(D,axis=0)
def logD2(d,N):#independent prior log(P(d_m,i|d_a)P(d_a|m,nm))
    da=np.sum(d,axis=0)
    C=stod([da])
    no=np.sum(d,axis=1)
    DS=np.sum(no)
    return (entD(C,N)+logq(sum(no),N)+DS*(log(DS)-1)-np.sum(logfac(da))-np.sum(logfac(no))+(0.5/(DS**2))*np.sum(da**2-da)*np.sum(no**2-no))

def logMxD(d,N):
    C,zeros=stodZero(d)
    return sum([log(min([N,np.sum(d[i])]))+logBI(N,zeros[i]) +min(entD([C[i]],N-zeros[i])+logq(np.sum(d[i])+zeros[i]-N,N-zeros[i]),logDB(N-zeros[i],np.sum(d[i])+zeros[i]-N)) for i in range(len(d)) if np.sum(d[i])])


def PDminO(vm,M,nm,N):
    Ds=[vmtoOS(N,vm[i],M[i]) for i in range(len(M))]
  
    return sum([logD1(Ds[i],N) for i in range(len(M))])




def PDminNO(vm,M,nm,N):
    Ds=[vmtoNOS(N,vm[i],M[i]) for i in range(len(M))]
    D1=sum([logD1(Ds[i],N) for i in range(len(M))])
    return sum([logD1(Ds[i],N) for i in range(len(M))])
def PDminA(Ds,M,nm,N):
    D1=logD1(Ds,N) 
    return D1
def PDminDAS(vm,M,nm,N):
    Ds=vmtoDAS(N,vm,M)
    return sum([logD1([Ds[i]],N) for i in range(3)])
#@jit                                                        
def vmtoOS(N,vm,m):#convert set of vertex maps to Orbit D. Sequence
    Dm=np.zeros((len(m.gp.orbits),N),dtype='int64')
    for i in vm:
        for v in range(m.num_vertices()):
            Dm[m.gp.orbmem[v]][i[v]]+=1        
    return Dm 
def vmtoNOS(N,vm,m):#convert set of vertex maps to Orbit D. Sequence
    Dm=np.zeros((1,N),dtype='int64')
    for i in vm:
        for v in range(m.num_vertices()):
            Dm[0][i[v]]+=1        
    return Dm  
def vmtoAS(N,VM,M):#convert set of vertex maps to Atomic D. Sequence
    Dm=np.zeros((1,N),dtype='int64')
    for m in range(len(M)):
        for i in VM[m]:
            for v in range(M[m].num_vertices()):
                Dm[0][i[v]]+=1        
    return Dm   
def vmtoDAS(N,VM,M):
    Dm=np.zeros((3,N),dtype='int64')
    orb=0
    for m in range(len(M)):
        for i in VM[m]:
            for v in range(M[m].num_vertices()):
                if M[m].vp.orbtype[v]==0:
                    Dm[0][i[v]]+=1
                elif M[m].vp.orbtype[v]==1:
                    Dm[1][i[v]]+=1
                elif M[m].vp.orbtype[v]==2:
                    Dm[2][i[v]]+=1

    return Dm

#motif priors 
def PMlogstar(M):#logstar prior
    return sum([logStar(m.gp.index) for m in M])-0.9#approximate normalizing constant
#Entropy functions
def entDCO(s,m,nm): #-log(C|d_m,o)
    return  (-logfac(nm)-nm*log(m.gp.hom)
            +sum([logfac(len(o)*nm) for o in m.gp.orbits])-np.sum(logfac(s))
            -m.gp.hom*(nm**2)*0.5*np.prod(np.array([(((np.sum(s[i]**2)/np.sum(s[i]))-1)/(nm*len(m.gp.orbits[i])))**len(m.gp.orbits[i]) for i in range(len(m.gp.orbits))]))
            -0.5*(m.num_vertices()*((np.sum(np.sum(s,axis=0)**2)/np.sum(np.sum(s,axis=0)))-1)
            -sum(np.array([(np.sum(i**2)/np.sum(i)-1) for i in s]))))
def entDCNO(s,m,nm): #-log(C|d_m)
    return (-logfac(nm)-nm*log(m.gp.hom)
            +logfac(m.num_vertices()*nm)-np.sum(logfac(s))
            -m.gp.hom*(nm**2)*0.5*(((np.sum(s[0]**2)/np.sum(s[0]))-1)*(1.0/(nm*m.num_vertices())))**m.num_vertices()
            -0.5*(m.num_vertices()-1)*((np.sum(s**2)/np.sum(s))-1))
def entDCA(s,M,nm): #-log(C|d,nm)
    NS=float(np.sum(s))
    return (logfac(NS)-np.sum(logfac(s))+sum([-logfac(nm[i])-nm[i]*log(M[i].gp.hom)                                              
            -M[i].gp.hom*(nm[i]**2)*0.5*(((np.sum(s**2)/NS)-1)*(1.0/(NS)))**M[i].num_vertices()                              
            -0.5*(M[i].num_vertices()-1)*M[i].num_vertices()*nm[i]*((np.sum(s**2)/NS)-1)/NS 
            for i in range(len(M))])
            )
def entDCDAS(s,M,nm):
    S=[np.sum(s[i]) for i in range(3)]
    S2=[np.sum(s[i]**2) for i in range(3)]
    ss2=np.sum(np.sum(s,axis=0)**2)
    ent=(-sum([nm[i]*log(M[i].gp.hom)+logfac(nm[i]) for i in range(len(M))])+np.sum(logfac(np.array([np.sum(s[i]) for i in range(3)])))-np.sum(logfac(s))
         -sum([M[i].gp.hom*0.5*(nm[i]**2)*prod([((S2[j]/S[j]-1)/(S[j]))**(M[i].gp.orbtc[j]) for j in range(3) if S[j]]) for i in range(len(M))])
         -0.5*sum([((M[i].num_vertices()**2)*nm[i]/np.sum(S))*(ss2/np.sum(S)-1)-sum([(nm[i]*M[i].gp.orbtc[j]/S[j])*(S2[j]/S[j]-1) for j in range(3) if S[j]]) for i in range(len(M))]))
    return ent
def HNm(m,n):
    t=1
    for i in range(m.num_vertices()):
        t=t*(n-i)
    return t//m.gp.hom
def entH(M,vm,N):#-logP(C|M,nm,N)
    return sum([logBID(HNm(M[i],N),len(vm[i])) for i in range(len(M))])
def entHMx(M,vm,N):#-logP(C|M,nm,N)
    zeros=[]
    for i in range(len(M)):
        s=vmtoNOS(N,vm[i],M[i])
        zeros.append(stodZero(s)[1][0])
    return sum([min(logBI(HNm(M[i],N-zeros[i]),len(vm[i]))+log(min([N,M[i].num_vertices()*len(vm[i])]))+logBI(N,zeros[i]),entH([M[i]],[vm[i]],N)) for i in range(len(M))])
def prod(a):
    p=1
    for i in a:
        p=p*i
    return p
#Description lengths
#Orbit degree model
def vmtoOS(N,vm,m):#convert set of vertex maps to Orbit D. Sequence
    Dm=np.zeros((len(m.gp.orbits),N),dtype='int')
    for i in vm:
        for v in range(m.num_vertices()):
            try:
                Dm[m.gp.orbmem[v]][i[v]]+=1
            except:
                print(m,m.gp.orbmem[v],i[v],N)
    return Dm
def SIGMAO(vm,M,N,E):#SIGMA for orbit degree model
    s=[vmtoOS(N,vm[i],M[i]) for i in range(len(M))]
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    return sum([entDCO(s[i],M[i],nm[i]) for i in range(len(M))])+ents(E,ems)+PMlogstar(M)+PDminO(vm,M,nm,N)
def SIGMAOI(vm,M,N,E):#SIGMA for orbit degree model -incomplete cover 
    s=[vmtoOS(N,vm[i],M[i]) for i in range(len(M))]
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    Em=sum([M[i].num_edges()*nm[i] for i in range(len(M))])
    return sum([entDCO(s[i],M[i],nm[i]) for i in range(len(M))])+ents(Em,ems)+PMlogstar(M)+PDminO(vm,M,nm,N)
def SIGMAO_D(vm,M,N,E):#detailled version of SIGMAO
    s=[vmtoOS(N,vm[i],M[i]) for i in range(len(M))]
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    
    return sum([entDCO(s[i],M[i],nm[i]) for i in range(len(M))])+ents(E,ems)+PMlogstar(M)+PDminO(vm,M,nm,N),PDminO(vm,M,nm,N),PMlogstar(M),ents(E,ems)+PMlogstar(M)+PDminO(vm,M,nm,N)[0],{M[i]:entDCO(s[i],M[i],nm[i]) for i in range(len(M))}  
def SIGMAO_E(vm,M,N,E,g,me):#SIGMA for orbit degree model for incomplete cover + uncovered single edges  
    vme=np.array([[g.vp.OI[e.source()],g.vp.OI[e.target()]] for e in g.edges()])
    return SIGMAO(vm+[vme],M+[me],N,E)
def sigmaO(g,m,vm,C):#efficiency of m in covering edges when added to C
    return (SIGMAOI(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E)-C[2])/(m.num_edges()*len(vm))
def DSIGMAO(g,m,vm,C):#Change in Sigma when m is added to C.
    if m.num_edges()==1:
        return -0.000002#in case m is the single edge
    else:
        return (SIGMAO_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[3])

#Orbit aggregated model
def vmtoNOS(N,vm,m):#convert set of vertex maps to Atomic D. Sequence
    Dm=np.zeros((1,N),dtype='int')
    for i in vm:
        for v in range(m.num_vertices()):
            Dm[0][i[v]]+=1        
    return Dm
def SIGMANO(vm,M,N,E):
    s=[vmtoNOS(N,vm[i],M[i]) for i in range(len(M))]
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    return sum([entDCNO(s[i],M[i],nm[i]) for i in range(len(M))])+ents(E,ems)+PMlogstar(M)+PDminNO(vm,M,nm,N)
def SIGMANO_E(vm,M,N,E,g,me):#SIGMA for orbit degree model for incomplete cover
    if g.num_edges():
        vme=np.array([[g.vp.OI[e.source()],g.vp.OI[e.target()]] for e in g.edges()])
        return SIGMANO(vm+[vme],M+[me],N,E)
    else:
        return SIGMANO(vm,M,N,E)
def SIGMANOI(vm,M,N,E):
    s=[vmtoNOS(N,vm[i],M[i]) for i in range(len(M))]
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    Em=sum([M[i].num_edges()*nm[i] for i in range(len(M))])
    return sum([entDCNO(s[i],M[i],nm[i]) for i in range(len(M))])+ents(Em,ems)+PMlogstar(M)+PDminNO(vm,M,nm,N)
def sigmaNOs(g,m,vm,C):
    if m.num_edges()==1:
        return -0.000002#in case m is the single edge
    else:
        return (SIGMANO_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[2])/(m.num_edges()*len(vm))
def sigmaNO(g,m,vm,C):
    return (SIGMANOI(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E)-C[2])/(m.num_edges()*len(vm))
def DSIGMANO(g,m,vm,C):
    if m.num_edges()==1:
        return -0.000002#in case m is the single edge
    else:
        return (SIGMANO_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[3])
#Total atomic degree model
def vmtoAS(N,VM,M):#convert set of vertex maps to Atomic D. Sequence
    Dm=np.zeros((1,N),dtype='int')
    for m in range(len(M)):
        for i in VM[m]:
            for v in range(M[m].num_vertices()):
                Dm[0][i[v]]+=1        
    return Dm 
def SIGMAA(vm,M,N,E):
    s=vmtoAS(N,vm,M)
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    return entDCA(s,M,nm)+ents(E,ems)+PMlogstar(M)+PDminA(s,M,nm,N)
def SIGMAAI(vm,M,N,E):
    s=vmtoAS(N,vm,M)
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    Em=sum([M[i].num_edges()*nm[i] for i in range(len(M))])
    return entDCA(s,M,nm)+ents(E,ems)+PMlogstar(M)+PDminA(s,M,nm,N)
def SIGMAA_E(vm,M,N,E,g,me):
    vme=np.array([[g.vp.OI[e.source()],g.vp.OI[e.target()]] for e in g.edges()])
    return SIGMAA(vm+[vme],M+[me],N,E)
def sigmaA(g,m,vm,C):
    return (SIGMAAI(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E)-C[2])/(m.num_edges()*len(vm))
def DSIGMAA(g,m,vm,C):
    if m.num_edges()==1:
        return -0.000002#in case m is the single edge
    else:
        return (SIGMAA_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[3])
def sigmaAs(g,m,vm,C):
    if m.num_edges()==1:
        return -0.000002#in case m is the single edge
    else:
        return (SIGMAA_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[2])/(m.num_edges()*len(vm))
#Homogeneous models
def SIGMAH(vm,M,N,E):
    return entH(M,vm,N)+ents(E,[m.num_edges() for m in M])+PMlogstar(M)
def SIGMAHI(vm,M,N,E):
    ems=[m.num_edges() for m in M]
    Em=sum([M[i].num_edges()*len(vm[i]) for i in range(len(M))])
    return entH(M,vm,N)+ents(Em,ems)+PMlogstar(M)
def SIGMAH_E(vm,M,N,E,g,me):
    vme=np.array([[g.vp.OI[e.source()],g.vp.OI[e.target()]] for e in g.edges()])
    return SIGMAH(vm+[vme],M+[me],N,E)
def SIGMAH_D(vm,M,N,E):
    s=entH(M,vm,N)+ents(E,[m.num_edges() for m in M])+PMlogstar(M)
    return s,entHMx(vm,M,N),PMlogstar(M)
def sigmaHs(g,m,vm,C):
    if m.num_edges()==1:
        return -0.00001
    else:
        return (SIGMAH_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[2])/(m.num_edges()*len(vm))
def sigmaH(g,m,vm,C):
    return (SIGMAHI(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E)-C[2])/(m.num_edges()*len(vm))
def DSIGMAH(g,m,vm,C):
    if m.num_edges()==1:
        return -0.00001
    else:
        return SIGMAH_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[3]
#Directed orbit degree model
def SIGMADAS(vm,M,N,E):#SIGMA for orbit degree model
    s=vmtoDAS(N,vm,M)
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    return entDCDAS(s,M,nm)+ents(E,ems)+PMlogstar(M)+PDminDAS(vm,M,nm,N)
def SIGMADASI(vm,M,N,E):#SIGMA for orbit degree model
    s=vmtoDAS(N,vm,M)
    nm=[len(vm[i]) for i in range(len(M))]
    ems=[m.num_edges() for m in M]
    Em=sum([M[i].num_edges()*nm[i] for i in range(len(M))])
    return entDCDAS(s,M,nm)+ents(E,ems)+PMlogstar(M)+PDminDAS(vm,M,nm,N)
def SIGMADAS_E(vm,M,N,E,g,me):#SIGMA for orbit degree model for incomplete cover
    vme=np.array([[g.vp.OI[e.source()],g.vp.OI[e.target()]] for e in g.edges()])
    return SIGMADAS(vm+[vme],M+[me],N,E)
def sigmaDASs(g,m,vm,C):
    if m.num_edges()==1:
        return -0.000002#m is the single edge
    else:
        return (SIGMADASI(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[2])/(m.num_edges()*len(vm))
def sigmaDAS(g,m,vm,C):
    return (SIGMADASI(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E)-C[2])/(m.num_edges()*len(vm))
def DSIGMADAS(g,m,vm,C):
    if m.num_edges()==1:
        return -0.000002#in case m is the single edge
    else:
        return (SIGMADAS_E(C[0]+[vm],C[1]+[m],g.num_vertices(),g.gp.E,g,g.gp.me)-C[3])

#inference for Orbit degree model
from multiprocessing import Pool, Manager
import itertools
def MINSF(g,m,C):#determine sigma-optimal set of m given partial cover 
    vm=[]
    em=[]
    mi=toig(m)
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    gi=toig(G)
    
    while True:
        s=gi.subisomorphic_lad(mi,return_mapping=True)
        if s[0]:
            se=[[s[1][G.vertex_index[e.source()]],s[1][G.vertex_index[e.target()]]] for e in m.edges()]
            vm.append([G.vp.OI[G.vertex(i)] for i in s[1]])
            em.append(se)
            for e in se:
                gi.delete_edges(gi.get_eid(e[0],e[1]))
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
    
    if len(vm)>1:     
        #sigmaO(vmC,MC,N,E,DM,me,g,m,vm,sigC)
        sm=sigmaO(G,m,vm,C)
        de=DSIGMAO(G,m,vm,C)
        #print(m,sm)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]

def MINSFgt(g,m,C):#determine sigma-optimal set of m-subgraphs given partial cover C
    vm=[]
    em=[]
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    while True:
        s=subgraph_isomorphism(m,G,max_n=1)
        if s:
            se=[(s[0][e.source()],s[0][e.target()]) for e in m.edges()]
            vm.append([G.vp.OI[s[0][v]] for v in m.vertices()])
            for e in se:
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
        
    if len(vm)>1:     
        sm=sigmaO(G,m,vm,C)
        de=DSIGMAO(G,m,vm,C)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
    
    
def SIGoptSFD(g,M,n,C):# find most efficient m
    p=Pool(n)
    p=get_context("fork").Pool(n)
    R=[p.apply_async(MINSF,(g,m,C)) for m in M]
    res=[r.get() for r in R]
    p.close()
    I=[i[1] for i in res]
    I2=[i[1] for i in res if i[3]<=0.0]
    print('Mininfo:',min(I2))
    minindex=I.index(min(I2))
    res[minindex][2].gp.inG=False
    Mnew=[m[2] for m in res if m[2].gp.inG and (not subgraph_isomorphism(res[minindex][2],m[2]))]
    return res[minindex],Mnew
def SIGoptSFDgt(g,M,n,C):# find most efficient m
    if __name__ == '__main__':
        p=get_context("fork").Pool(n)
        R=[p.apply_async(MINSFgt,(g,m,C)) for m in M]
        res=[r.get() for r in R]
        p.close()
        I=[i[1] for i in res]
        I2=[i[1] for i in res if i[3]<=0.0]
        print('Mininfo:',min(I2))
        minindex=I.index(min(I2))
        res[minindex][2].gp.inG=False
        Mnew=[m[2] for m in res if m[2].gp.inG and (not subgraph_isomorphism(res[minindex][2],m[2]))]
    return res[minindex],Mnew
def SIGoptCFD(g,M,n):
    m0=M[0].copy()
    rd.shuffle(M)
    me=g.new_gp('object',val=m0)
    g.gp.me=me
    ed=g.new_graph_property("int",val=g.num_edges())
    g.gp.E=ed
    oivp = g.new_vertex_property("int",vals=g.vertex_index.copy())
    g.vp.OI=oivp
    cove=g.new_edge_property("bool",val=True)
    g.ep.covered=cove
    g.set_edge_filter(g.ep.covered)
    vorder = g.new_vertex_property("int")
    g.vp.vorder=vorder
    C=[[],[],0.0,SIGMAO_E([],[],g.num_vertices(),g.gp.E,g,g.gp.me)]
    print('Sigma0:',C[3])
    while g.num_edges():
        r=[]
        for v in g.vertices():
            r.append([v,v.out_degree()+v.in_degree()])
        r.sort(key=lambda d:d[1])
        for i in range(len(r)):
            g.vp.vorder[r[i][0]]=i
        G=gt.Graph(g,vorder=vorder)
        mopt,MM=SIGoptSFD(G,M,n,C)
        M=MM
        print('Motifs Left:',len(M))
        for vm in mopt[0]:
            for e in mopt[2].edges():
                g.ep.covered[g.edge(vm[mopt[2].vertex_index[e.source()]],vm[mopt[2].vertex_index[e.target()]])]=False
        C[1].append(mopt[2])#atoms
        C[0].append(mopt[0])#vertex maps
        if g.num_edges():
            C[2]=SIGMAOI(C[0],C[1],g.num_vertices(),g.gp.E)
            C[3]=SIGMAO_E(C[0],C[1],g.num_vertices(),g.gp.E,g,g.gp.me)
            print('SIGMA:',C[3],C[2],mopt[2],len(mopt[0]))
    print('SIGMA Final:',SIGMAO(C[0],C[1],g.num_vertices(),g.gp.E))
    return C
def SIGoptCFDgt(g,M,n):
    m0=M[0].copy()
    rd.shuffle(M)
    me=g.new_gp('object',val=m0)
    g.gp.me=me
    ed=g.new_graph_property("int",val=g.num_edges())
    g.gp.E=ed
    oivp = g.new_vertex_property("int",vals=g.vertex_index.copy())
    g.vp.OI=oivp
    cove=g.new_edge_property("bool",val=True)
    g.ep.covered=cove
    g.set_edge_filter(g.ep.covered)
    vorder = g.new_vertex_property("int")
    g.vp.vorder=vorder
    C=[[],[],0.0,SIGMAO_E([],[],g.num_vertices(),g.gp.E,g,g.gp.me)]
    print('Sigma0:',C[3])
    while g.num_edges():
        r=[]
        for v in g.vertices():
            r.append([v,v.out_degree()+v.in_degree()])
        r.sort(key=lambda d:d[1])
        for i in range(len(r)):
            g.vp.vorder[r[i][0]]=i
        G=gt.Graph(g,vorder=vorder)
        mopt,MM=SIGoptSFD(G,M,n,C)
        M=MM
        print('Motifs Left:',len(M))
        for vm in mopt[0]:
            for e in mopt[2].edges():
                g.ep.covered[g.edge(vm[mopt[2].vertex_index[e.source()]],vm[mopt[2].vertex_index[e.target()]])]=False
        C[1].append(mopt[2])#atoms
        C[0].append(mopt[0])#vertex maps
        if g.num_edges():
            C[2]=SIGMAOI(C[0],C[1],g.num_vertices(),g.gp.E)
            C[3]=SIGMAO_E(C[0],C[1],g.num_vertices(),g.gp.E,g,g.gp.me)
            print('SIGMA:',C[3],C[2],mopt[2],len(mopt[0]))
    print('SIGMA Final:',SIGMAO(C[0],C[1],g.num_vertices(),g.gp.E))
    return C
#inference for Directed orbits model
from multiprocessing import Pool, Manager
import itertools
def MINSFDASgt(g,m,C):#determine sigma-optimal set of m-subgraphs given partial cover C
    vm=[]
    em=[]
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    while True:
        s=subgraph_isomorphism(m,G,max_n=1)
        if s:
            se=[(s[0][e.source()],s[0][e.target()]) for e in m.edges()]
            vm.append([G.vp.OI[s[0][v]] for v in m.vertices()])
            for e in se:
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
        
    if len(vm)>1:     
        sm=sigmaDAS(G,m,vm,C)
        de=DSIGMADAS(G,m,vm,C)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
def MINSFDAS(g,m,C):#determine sigma-optimal set of m given partial cover 
    vm=[]
    em=[]
    mi=toig(m)
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    gi=toig(G)
    
    while True:
        s=gi.subisomorphic_lad(mi,return_mapping=True)
        if s[0]:
            se=[[s[1][G.vertex_index[e.source()]],s[1][G.vertex_index[e.target()]]] for e in m.edges()]
            vm.append([G.vp.OI[G.vertex(i)] for i in s[1]])
            em.append(se)
            for e in se:
                gi.delete_edges(gi.get_eid(e[0],e[1]))
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
    
    if len(vm)>1:     
        #sigmaO(vmC,MC,N,E,DM,me,g,m,vm,sigC)
        sm=sigmaDAS(G,m,vm,C)
        de=DSIGMADAS(G,m,vm,C)
        #print(m,sm)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
def SIGoptSFDAS(g,M,n,C):# find most efficient m
    p=Pool(n)
    R=[p.apply_async(MINSFDAS,(g,m,C)) for m in M]
    res=[r.get() for r in R]
    p.close()
    I=[i[1] for i in res]
    I2=[i[1] for i in res if i[3]<0.0]
    print('Mininfo:',min(I2))
    minindex=I.index(min(I2))
    res[minindex][2].gp.inG=False
    Mnew=[m[2] for m in res if m[2].gp.inG and (not subgraph_isomorphism(res[minindex][2],m[2]))]
    return res[minindex],Mnew

def SIGoptCFDAS(g,M,n):
    for m in M:
        orbtype=m.new_vertex_property('int')
        m.vp.orbtype=orbtype
        orbtc=m.new_gp("vector<int>",val=[0,0,0])
        m.gp.orbtc=orbtc
        for v in m.vertices():
            if v.out_degree() and v.in_degree():
                m.gp.orbtc[2]+=1
                m.vp.orbtype[v]=2
            elif v.out_degree():
                m.vp.orbtype[v]=0
                m.gp.orbtc[0]+=1
            elif v.in_degree():
                m.vp.orbtype[v]=1
                m.gp.orbtc[1]+=1
    m0=M[0].copy()
    rd.shuffle(M)
    me=g.new_gp('object',val=m0)
    g.gp.me=me
    ed=g.new_graph_property("int",val=g.num_edges())
    g.gp.E=ed
    oivp = g.new_vertex_property("int",vals=g.vertex_index.copy())
    g.vp.OI=oivp
    cove=g.new_edge_property("bool",val=True)
    g.ep.covered=cove
    g.set_edge_filter(g.ep.covered)
    vorder = g.new_vertex_property("int")
    g.vp.vorder=vorder
    C=[[],[],0.0,SIGMADAS_E([],[],g.num_vertices(),g.gp.E,g,g.gp.me)]
    print('Sigma0:',C[3])
    while g.num_edges():
        r=[]
        for v in g.vertices():
            r.append([v,v.out_degree()])
        r.sort(key=lambda d:d[1])
        for i in range(len(r)):
            g.vp.vorder[r[i][0]]=i
        G=gt.Graph(g,vorder=vorder)
        mopt,MM=SIGoptSFDAS(G,M,n,C)
        M=MM
        print('Motifs Left:',len(M))
        for vm in mopt[0]:
            for e in mopt[2].edges():
                g.ep.covered[g.edge(vm[mopt[2].vertex_index[e.source()]],vm[mopt[2].vertex_index[e.target()]])]=False
        C[1].append(mopt[2])#atoms
        C[0].append(mopt[0])#vertex maps
        if g.num_edges():
            C[2]=SIGMADASI(C[0],C[1],g.num_vertices(),g.gp.E)
            C[3]=SIGMADAS_E(C[0],C[1],g.num_vertices(),g.gp.E,g,g.gp.me)
            print('SIGMA:',C[3],C[2],mopt[2],len(mopt[0]))

    print('SIGMADAS Final:',SIGMADAS(C[0],C[1],g.num_vertices(),g.gp.E))
    return C

#inference for Orbit aggregated degree model
from multiprocessing import Pool, Manager
import itertools
def MINSFNOgt(g,m,C):#determine sigma-optimal set of m-subgraphs given partial cover C
    vm=[]
    em=[]
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    while True:
        s=subgraph_isomorphism(m,G,max_n=1)
        if s:
            se=[(s[0][e.source()],s[0][e.target()]) for e in m.edges()]
            vm.append([G.vp.OI[s[0][v]] for v in m.vertices()])
            for e in se:
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
        
    if len(vm)>1:     
        sm=sigmaNO(G,m,vm,C)
        de=DSIGMANO(G,m,vm,C)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
    
def MINSFNO(g,m,C):#determine sigma-optimal set of m given partial cover 
    vm=[]
    em=[]
    mi=toig(m)
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    gi=toig(G)
    
    while True:
        s=gi.subisomorphic_lad(mi,return_mapping=True)
        if s[0]:
            se=[[s[1][G.vertex_index[e.source()]],s[1][G.vertex_index[e.target()]]] for e in m.edges()]
            vm.append([G.vp.OI[G.vertex(i)] for i in s[1]])
            em.append(se)
            for e in se:
                gi.delete_edges(gi.get_eid(e[0],e[1]))
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
    
    if len(vm)>1:     
        #sigmaO(vmC,MC,N,E,DM,me,g,m,vm,sigC)
        sm=sigmaNO(G,m,vm,C)
        de=DSIGMANO(G,m,vm,C)
        #print(m,sm)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
def SIGoptSFDNO(g,M,n,C):# find most efficient m
    p=Pool(n)
    R=[p.apply_async(MINSFNO,(g,m,C)) for m in M]
    res=[r.get() for r in R]
    p.close()
    I=[i[1] for i in res]
    I2=[i[1] for i in res if i[3]<=0.0]
    print('Mininfo:',min(I2))
    minindex=I.index(min(I2))
    res[minindex][2].gp.inG=False
    Mnew=[m[2] for m in res if m[2].gp.inG and (not subgraph_isomorphism(res[minindex][2],m[2]))]
    return res[minindex],Mnew

def SIGoptCFDNO(g,M,n):
    m0=M[0].copy()
    rd.shuffle(M)
    me=g.new_gp('object',val=m0)
    g.gp.me=me
    ed=g.new_graph_property("int",val=g.num_edges())
    g.gp.E=ed
    oivp = g.new_vertex_property("int",vals=g.vertex_index.copy())
    g.vp.OI=oivp
    cove=g.new_edge_property("bool",val=True)
    g.ep.covered=cove
    g.set_edge_filter(g.ep.covered)
    vorder = g.new_vertex_property("int")
    g.vp.vorder=vorder
    C=[[],[],0.0,SIGMANO_E([],[],g.num_vertices(),g.gp.E,g,g.gp.me)]
    print('Sigma0:',C[3])
    while g.num_edges():
        r=[]
        for v in g.vertices():
            r.append([v,v.out_degree()+v.in_degree()])
        r.sort(key=lambda d:d[1])
        for i in range(len(r)):
            g.vp.vorder[r[i][0]]=i
        G=gt.Graph(g,vorder=vorder)
        mopt,MM=SIGoptSFDNO(G,M,n,C)
        M=MM
        print('Motifs Left:',len(M))
        for vm in mopt[0]:
            for e in mopt[2].edges():
                g.ep.covered[g.edge(vm[mopt[2].vertex_index[e.source()]],vm[mopt[2].vertex_index[e.target()]])]=False
        C[1].append(mopt[2])#atoms
        C[0].append(mopt[0])#vertex maps
        if g.num_edges():
            C[2]=SIGMANOI(C[0],C[1],g.num_vertices(),g.gp.E)
            C[3]=SIGMANO_E(C[0],C[1],g.num_vertices(),g.gp.E,g,g.gp.me)
            print('SIGMA:',C[3],C[2],mopt[2],len(mopt[0]))
    print('SIGMA Final:',SIGMANO(C[0],C[1],g.num_vertices(),g.gp.E))
    return C

#inference for Orbit degree model
from multiprocessing import Pool, Manager
import itertools
def MINSFAgt(g,m,C):#determine sigma-optimal set of m-subgraphs given partial cover C
    vm=[]
    em=[]
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    while True:
        s=subgraph_isomorphism(m,G,max_n=1)
        if s:
            se=[(s[0][e.source()],s[0][e.target()]) for e in m.edges()]
            vm.append([G.vp.OI[s[0][v]] for v in m.vertices()])
            for e in se:
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
        
    if len(vm)>1:     
        sm=sigmaA(G,m,vm,C)
        de=DSIGMAA(G,m,vm,C)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
def MINSFA(g,m,C):#determine sigma-optimal set of m given partial cover 
    vm=[]
    em=[]
    mi=toig(m)
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    gi=toig(G)
    
    while True:
        s=gi.subisomorphic_lad(mi,return_mapping=True)
        if s[0]:
            se=[[s[1][G.vertex_index[e.source()]],s[1][G.vertex_index[e.target()]]] for e in m.edges()]
            vm.append([G.vp.OI[G.vertex(i)] for i in s[1]])
            em.append(se)
            for e in se:
                gi.delete_edges(gi.get_eid(e[0],e[1]))
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
    
    if len(vm)>1:     
        #sigmaO(vmC,MC,N,E,DM,me,g,m,vm,sigC)
        sm=sigmaA(G,m,vm,C)
        de=DSIGMAA(G,m,vm,C)
        #print(m,sm)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]
    
    
def SIGoptSFDA(g,M,n,C):# find most efficient m
    
    p=Pool(n)
    R=[p.apply_async(MINSFA,(g,m,C)) for m in M]
    res=[r.get() for r in R]
    p.close()
    I=[i[1] for i in res]
    I2=[i[1] for i in res if i[3]<=0.0]
    print('Mininfo:',min(I2))
    minindex=I.index(min(I2))
    res[minindex][2].gp.inG=False
    Mnew=[m[2] for m in res if m[2].gp.inG and (not subgraph_isomorphism(res[minindex][2],m[2]))]
    return res[minindex],Mnew

def SIGoptCFDA(g,M,n):
    m0=M[0].copy()
    rd.shuffle(M)
    me=g.new_gp('object',val=m0)
    g.gp.me=me
    ed=g.new_graph_property("int",val=g.num_edges())
    g.gp.E=ed
    oivp = g.new_vertex_property("int",vals=g.vertex_index.copy())
    g.vp.OI=oivp
    cove=g.new_edge_property("bool",val=True)
    g.ep.covered=cove
    g.set_edge_filter(g.ep.covered)
    vorder = g.new_vertex_property("int")
    g.vp.vorder=vorder
    C=[[],[],0.0,SIGMAA_E([],[],g.num_vertices(),g.gp.E,g,g.gp.me)]
    print('Sigma0:',C[3])
    while g.num_edges():
        r=[]
        for v in g.vertices():
            r.append([v,v.out_degree()])
        r.sort(key=lambda d:d[1])
        for i in range(len(r)):
            g.vp.vorder[r[i][0]]=i
        G=gt.Graph(g,vorder=vorder)
        mopt,MM=SIGoptSFDA(G,M,n,C)
        M=MM
        print('Motifs Left:',len(M))
        for vm in mopt[0]:
            for e in mopt[2].edges():
                g.ep.covered[g.edge(vm[mopt[2].vertex_index[e.source()]],vm[mopt[2].vertex_index[e.target()]])]=False
        C[1].append(mopt[2])#atoms
        C[0].append(mopt[0])#vertex maps
        if g.num_edges():
            C[2]=SIGMAAI(C[0],C[1],g.num_vertices(),g.gp.E)
            C[3]=SIGMAA_E(C[0],C[1],g.num_vertices(),g.gp.E,g,g.gp.me)
            print('SIGMA:',C[3],C[2],mopt[2],len(mopt[0]))
    print('SIGMA Final:',SIGMAA(C[0],C[1],g.num_vertices(),g.gp.E))
    return C


#inference homogeneoush models
from multiprocessing import Pool, Manager
import itertools
def MINSFHgt(g,m,C):#determine sigma-optimal set of m-subgraphs given partial cover C
    vm=[]
    em=[]
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    while True:
        s=subgraph_isomorphism(m,G,max_n=1)
        if s:
            se=[(s[0][e.source()],s[0][e.target()]) for e in m.edges()]
            vm.append([G.vp.OI[s[0][v]] for v in m.vertices()])
            for e in se:
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
        
    if len(vm)>1:     
        sm=sigmaH(G,m,vm,C)
        de=DSIGMAH(G,m,vm,C)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]

def MINSFH(g,m,C):#determine sigma-optimal set of m given partial cover 
    vm=[]
    em=[]
    mi=toig(m)
    G=gt.Graph(g)
    G.set_edge_filter(G.ep.covered)
    gi=toig(G)
    while True:
        s=gi.subisomorphic_lad(mi,return_mapping=True)
        if s[0]:
            se=[[s[1][G.vertex_index[e.source()]],s[1][G.vertex_index[e.target()]]] for e in m.edges()]
            vm.append([G.vp.OI[G.vertex(i)] for i in s[1]])
            em.append(se)
            for e in se:
                gi.delete_edges(gi.get_eid(e[0],e[1]))
                G.ep.covered[G.edge(e[0],e[1])]=False
        else:
            break
    if len(vm)>1:     
        #sigmaO(vmC,MC,N,E,DM,me,g,m,vm,sigC)
        sm=sigmaH(G,m,vm,C)
        de=DSIGMAH(G,m,vm,C)
        #print(m,sm)
        m.gp.inf=sm
        return [vm,m.gp.inf,m,de]
    else:
        m.gp.inG=False
        return [[],10000,m,1.0]    
    
def SIGoptSFDH(g,M,n,C):# find most efficient m
    p=Pool(n)
    R=[p.apply_async(MINSFH,(g,m,C)) for m in M]
    res=[r.get() for r in R]
    p.close()
    I=[i[1] for i in res]
    I2=[i[1] for i in res if i[3]<=0.0]
    print('Mininfo:',min(I2))
    minindex=I.index(min(I2))
    res[minindex][2].gp.inG=False
    Mnew=[m[2] for m in res if m[2].gp.inG and (not subgraph_isomorphism(res[minindex][2],m[2]))]
    return res[minindex],Mnew

def SIGoptCFDH(g,M,n):
    m0=M[0].copy()
    rd.shuffle(M)
    me=g.new_gp('object',val=m0)
    g.gp.me=me
    ed=g.new_graph_property("int",val=g.num_edges())
    g.gp.E=ed
    oivp = g.new_vertex_property("int",vals=g.vertex_index.copy())
    g.vp.OI=oivp
    cove=g.new_edge_property("bool",val=True)
    g.ep.covered=cove
    g.set_edge_filter(g.ep.covered)
    vorder = g.new_vertex_property("int")
    g.vp.vorder=vorder
    C=[[],[],0.0,SIGMAH_E([],[],g.num_vertices(),g.gp.E,g,g.gp.me)]
    print('Sigma0:',C[3])
    while g.num_edges():
        r=[]
        for v in g.vertices():
            r.append([v,v.out_degree()+v.in_degree()])
        r.sort(key=lambda d:d[1])
        for i in range(len(r)):
            g.vp.vorder[r[i][0]]=i
        G=gt.Graph(g,vorder=vorder)
        mopt,MM=SIGoptSFDH(G,M,n,C)
        M=MM
        print('Motifs Left:',len(M))
        for vm in mopt[0]:
            for e in mopt[2].edges():
                g.ep.covered[g.edge(vm[mopt[2].vertex_index[e.source()]],vm[mopt[2].vertex_index[e.target()]])]=False
        C[1].append(mopt[2])#atoms
        C[0].append(mopt[0])#vertex maps
        if g.num_edges():
            C[2]=SIGMAHI(C[0],C[1],g.num_vertices(),g.gp.E)
            C[3]=SIGMAH_E(C[0],C[1],g.num_vertices(),g.gp.E,g,g.gp.me)
            print('SIGMA:',C[3],C[2],mopt[2],len(mopt[0]))
    print('SIGMA Final:',SIGMAH(C[0],C[1],g.num_vertices(),g.gp.E))
    return C

def InferC(model,g,M,n):
    remove_parallel_edges(g)
    remove_self_loops(g)
    print(g)
    i=1
    for m in M:
        mi=m.new_gp('int',val=i)
        m.gp.index=mi
        m.gp.inf=-10000
        m.gp.index=i
        i+=1
    if model=='H': # nonDC/homogeneous model        
        CO=SIGoptCFDH(g,M,n)
    elif model=='T': # total degree model
        CO=SIGoptCFDA(g,M,n)
    elif model=='O': # orbit degree model       
        CO=SIGoptCFD(g,M,n)
    elif model=='M': # motif degree model
        CO=SIGoptCFDNO(g,M,n)
    elif model=='D': # directed degree model only for directed nets 
        CO=SIGoptCFDAS(g,M,n)
    return CO