# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import sys
import math
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# ------------------------ 


   
def normalisation(df):
    colonnes = df.columns.values
    
    # Passer du dataframe à des arrays:
    mat = np.array(df[colonnes])

    list_min = mat.min(axis=0)
    list_max = mat.max(axis=0)
    #calcule la différence entre les valeurs maximales et minimales pour chaque colonne 
    diff = list_max - list_min
    #normaliser la matrice
    sub = np.subtract(mat, list_min)
    mat_norm = np.divide(sub, diff)
    
    # Créer un nouveau DataFrame avec les mêmes indices que la DataFrame d'origine
    new_df = pd.DataFrame(mat_norm, columns=colonnes, index=df.index.copy())
    
    return new_df

def dist_euclidienne(v1,v2):
    """input : deux vecteurs ou ndarray
    output : float correspondant à distance euclidienne"""
    return np.linalg.norm(np.array(v1)-np.array(v2))

def centroide(v):
    """input : un vecteur ou ndarray
    output : un vecteur ou ndarray"""
    return np.mean(v,axis=0)

def dist_centroides(v1,v2):
    """input : deux vecteurs ou ndarray
    output : float correspondant à distance centroides entre v1 et v2"""
    c1=centroide(v1)
    c2=centroide(v2)
    return dist_euclidienne(c1,c2)

def dist_complete(v1, v2):
    """input : deux vecteurs ou ndarray
    output : float correspondant à distance complete entre v1 et v2"""
    w1=np.array(v1)
    w2=np.array(v2)
    max_dist=-1
    for e in w1:
        #calcule la distance euclidienne entre chaque ligne du tableau w2 et le vecteur e
        mat=np.sqrt(np.sum((w2-e)**2,axis=1))
        d=np.max(mat)
        if(d>max_dist):
            max_dist=d
    return max_dist
   
    
def dist_simple(v1,v2):
    """input : deux vecteurs ou ndarray
    output : float correspondant à distance simple entre v1 et v2"""
    w1=np.array(v1)
    w2=np.array(v2)
    min_dist=sys.float_info.max
    for e in w1:
        #calcule la distance euclidienne entre chaque ligne du tableau w2 et le vecteur e
        mat=np.sqrt(np.sum((w2-e)**2,axis=1))
        d=np.min(mat)
        if(d<min_dist):
            min_dist=d
    return min_dist


def dist_average(v1,v2):
    """input : deux vecteurs ou ndarray
    output : float correspondant à distance average entre v1 et v2"""
    w1=np.array(v1)
    w2=np.array(v2)
    
    somme=0
    for e in w1:
        #calcule la distance euclidienne entre chaque ligne du tableau w2 et le vecteur e
        mat=np.sqrt(np.sum((w2-e)**2,axis=1))
        somme+=np.sum(mat)
        
       
    return somme/(len(v1)*len(v2))


def inertie_cluster(Ens):
    """Ens est un dataFrame 
    calcule l'inertie d'un cluster
    """
    v_centroide=np.array(centroide(Ens))
    matrice=np.array(Ens)
    
    #somme des carrés des distances entre chaque vecteur du cluster et le centroïde 
    return np.sum(np.sum( (matrice-v_centroide)**2,axis=1) )  

def inertie_globale(Base, U):
    """input : dataframe * dictionnaire qui représente la matrice d'affectation
    output : float inertie global"""
    somme=0
    for key,value in U.items():
        somme+=inertie_cluster(Base.iloc[value])
    return somme

def init_kmeans(K,Ens):
    
    """
    input : int * dataframe
    renvoie un np.array composés de 𝐾 exemples tirés aléatoirement dans la base
    """
    indices_aleatoires = np.random.randint(0, len(Ens), size=K)
    
    return np.array(Ens.iloc[indices_aleatoires])
    
def plus_proche(Exe,Centres):
    """rend l'indice du centroide dont l'exemple est le plus proche
    """
    distances=np.sum( (np.array(Centres)-np.array(Exe))**2,axis=1)
    return np.argmin(distances)

def affecte_cluster(Base,Centres):
    """renvoie un dictionnaire pour représenter une matrice d'affectation
    """
    mat_affect={key: [] for key in range(len(Centres))}
    for i in range (len(Base)):
        ind_centre=plus_proche(Base.iloc[i],Centres)
        mat_affect[ind_centre].append(i)

    return mat_affect

def nouveaux_centroides(Base,U):
    """
    input : dataframe * dictionnaire qui représente la matrice d'affectation
    rend l'ensemble des nouveaux centroides
    """
    nv_centriodes=[]
    
    for key,value in U.items():
        nv_centriodes.append(centroide(Base.iloc[value]))
    
    return np.array(nv_centriodes)

def kmoyennes(K, Base, epsilon, iter_max):
    """
    input : K int (nombre de cluster souhaité) 
            Base dataframe (base d'apprentissage)
            epsilon > 0 float (critère d'arret)
            iter_max > 1 nombre d'itération max
    output : ensemble de centroides et une matrice d'affectation
    """

    centroides=init_kmeans(K,Base)
    inertie0=sys.float_info.max
    
    i=0
    seuil=10000
    
    while(seuil>epsilon and i< iter_max):
        affectation=affecte_cluster(Base,centroides)
        inertie1=inertie_globale(Base,affectation)
        seuil=abs(inertie1-inertie0)
        inertie0=inertie1
        centroides=nouveaux_centroides(Base,affectation)
        i+=1
    
    return centroides,affectation

def codistance_Ck(Ck):
    """ calcule la codistance maximale entre les points d'un cluster"""
    max=sys.float_info.min
    matr=np.array(Ck)
    
    for i in range (len(Ck)): 
        distances=np.sum((matr-matr[i])**2,axis=1)
        d=np.max(distances)
        if(d>max):     
            max=d
            
    return max 

def codistance_totale(base,Affectation):
    """calcule la codistance totale d'une dataframe (base) séparé en clusters (Affectation) """
    somme=0
    for cle, valeur in Affectation.items():
        somme+=codistance_Ck(base.iloc[valeur])
    
    return somme

def semin(Base,Affectation):
    """calcule la distance minimale entre les centroides des clusters (Affectation) 
    dans une base d'apprentissage (Base) et renvoie cette valeur"""
    centroides=nouveaux_centroides(Base,Affectation)
    min=sys.float_info.max
    
    for c in centroides: 
        distances=np.sum((centroides-c)**2,axis=1)
        indices=np.nonzero(distances)
        distances=distances[indices]
        d=np.min(distances)
        if(d<min):     
            min=d
            
    return min

def index_dunn(Base,Affectation):
    """on cherche à le minimiser"""
    return codistance_totale(Base,Affectation)/semin(Base,Affectation)

def index_XieBeni(Base,Affectation):
    """on cherche à le minimiser"""
    return inertie_globale (Base,Affectation)/semin(Base,Affectation)
 

def initialise_CHA(df):
    """input : dataframe 
    output : initialisation de CHA (chaque exemple est son propre cluster) """
    res=dict()
    for i in range(len(df)):
        res[i]=[i]
    return res



def fusionneCentroid(df,p0,verbose=False):
    """input : df est la base d'apprentissage
               po est un dictionnaire contenant les affectations actuelles des points aux clusters
               verbose si on veut afficher les distances trouvées entre deux clusters au fur et a mesure
    output : tuple contenant:
                            le nouveau dictionnaire
                            les clés des clusters fusionnés
                            la distance minimale entre les centroïdes des clusters fusionnés
    """
    cle1=0
    cle2=0
    dMin=sys.float_info.max
    for cle in p0:
        l1=np.array(p0[cle])
        d1=df.iloc[l1]
        for key in p0:
            if(cle!=key):
                l2=np.array(p0[key])
                d2=df.iloc[l2]
                dist=dist_centroides(d1,d2)
                if(dist<dMin):
                    cle1=cle
                    cle2=key
                    dMin=dist
    p1=dict()
    for cle in p0:
        if(cle==cle1 or cle==cle2):
            continue
        else:
            p1[cle]=p0[cle]
            
    keys=[cle for cle in p0.keys()]
    
    p1[np.max(keys)+1]=p0[cle1]+p0[cle2]
    if(verbose):
        print("Distance mininimale trouvée entre [",cle1,",",cle2,"] = ",dMin)
    
    return (p1,cle1,cle2,dMin)
        


def fusionneComplete(df,p0,verbose=False):
    """input : df est la base d'apprentissage
               po est un dictionnaire contenant les affectations actuelles des points aux clusters
               verbose si on veut afficher les distances trouvées entre deux clusters au fur et a mesure
    output : tuple contenant:
                            le nouveau dictionnaire
                            les clés des clusters fusionnés
                            la distance minimale entre les centroïdes des clusters fusionnés
    """
    cle1=0
    cle2=0
    dMin=sys.float_info.max
    for cle in p0:
        l1=np.array(p0[cle])
        d1=df.iloc[l1]
        for key in p0:
            if(cle!=key):
                l2=np.array(p0[key])
                d2=df.iloc[l2]
                dist=dist_complete(d1,d2)
                if(dist<dMin):
                    cle1=cle
                    cle2=key
                    dMin=dist
    p1=dict()
    for cle in p0:
        if(cle==cle1 or cle==cle2):
            continue
        else:
            p1[cle]=p0[cle]
            
    keys=[cle for cle in p0.keys()]
    
    p1[np.max(keys)+1]=p0[cle1]+p0[cle2]
    if(verbose):
        print("Distance mininimale trouvée entre [",cle1,",",cle2,"] = ",dMin)
    
    return (p1,cle1,cle2,dMin)
        


def fusionneAverage(df,p0,verbose=False):
    """input : df est la base d'apprentissage
               po est un dictionnaire contenant les affectations actuelles des points aux clusters
               verbose si on veut afficher les distances trouvées entre deux clusters au fur et a mesure
    output : tuple contenant:
                            le nouveau dictionnaire
                            les clés des clusters fusionnés
                            la distance minimale entre les centroïdes des clusters fusionnés
    """
    cle1=0
    cle2=0
    dMin=sys.float_info.max
    for cle in p0:
        l1=np.array(p0[cle])
        d1=df.iloc[l1]
        for key in p0:
            if(cle!=key):
                l2=np.array(p0[key])
                d2=df.iloc[l2]
                dist=dist_average(d1,d2)
                if(dist<dMin):
                    cle1=cle
                    cle2=key
                    dMin=dist
    p1=dict()
    for cle in p0:
        if(cle==cle1 or cle==cle2):
            continue
        else:
            p1[cle]=p0[cle]
            
    keys=[cle for cle in p0.keys()]
    
    p1[np.max(keys)+1]=p0[cle1]+p0[cle2]
    if(verbose):
        print("Distance mininimale trouvée entre [",cle1,",",cle2,"] = ",dMin)
    
    return (p1,cle1,cle2,dMin)



def fusionneSimple(df,p0,verbose=False):
    """input : df est la base d'apprentissage
               po est un dictionnaire contenant les affectations actuelles des points aux clusters
               verbose si on veut afficher les distances trouvées entre deux clusters au fur et a mesure
    output : tuple contenant:
                            le nouveau dictionnaire
                            les clés des clusters fusionnés
                            la distance minimale entre les centroïdes des clusters fusionnés
    """
    cle1=0
    cle2=0
    dMin=sys.float_info.max
    for cle in p0:
        l1=np.array(p0[cle])
        d1=df.iloc[l1]
        for key in p0:
            if(cle!=key):
                l2=np.array(p0[key])
                d2=df.iloc[l2]
                dist=dist_simple(d1,d2)
                if(dist<dMin):
                    cle1=cle
                    cle2=key
                    dMin=dist
    p1=dict()
    for cle in p0:
        if(cle==cle1 or cle==cle2):
            continue
        else:
            p1[cle]=p0[cle]
            
    keys=[cle for cle in p0.keys()]
    
    p1[np.max(keys)+1]=p0[cle1]+p0[cle2]
    if(verbose):
        print("Distance mininimale trouvée entre [",cle1,",",cle2,"] = ",dMin)
    
    return (p1,cle1,cle2,dMin)



def CHA_centroid(df,verbose=False,dendrogramme=False):
    """ dendrogramme centroid de df"""
    res=[]
    depart=initialise_CHA(df)
    p0,cle2,cle1,d=fusionneCentroid(df,depart)
    if(verbose):
        print("Clustering hiérarchique ascendant, version Centroid Linkage")
        print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
    tmp=dict(p0)
    last_key, last_value = tmp.popitem()
    somme=len(last_value)
    l=[cle2,cle1,d,somme]
    res.append(l)
    while(len(p0)>1):
        p0,cle2,cle1,d=fusionneCentroid(df,p0)
        if(verbose):
            print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
        tmp=dict(p0)
        last_key, last_value = tmp.popitem()
        somme=len(last_value)
        l=[cle2,cle1,d,somme]
        res.append(l)
    if(dendrogramme):
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return res



def CHA_complete(df,verbose=False,dendrogramme=False):
    """ dendrogramme complete de df"""
    res=[]
    depart=initialise_CHA(df)
    p0,cle2,cle1,d=fusionneComplete(df,depart)
    if(verbose):
        print("Clustering hiérarchique ascendant, version Complete Linkage")
        print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
    tmp=dict(p0)
    last_key, last_value = tmp.popitem()
    somme=len(last_value)
    l=[cle2,cle1,d,somme]
    res.append(l)
    while(len(p0)>1):
        p0,cle2,cle1,d=fusionneComplete(df,p0)
        if(verbose):
            print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
        tmp=dict(p0)
        last_key, last_value = tmp.popitem()
        somme=len(last_value)
        l=[cle2,cle1,d,somme]
        res.append(l)
    if(dendrogramme):
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return res



def CHA_average(df,verbose=False,dendrogramme=False):
    """ dendrogramme average de df"""
    res=[]
    depart=initialise_CHA(df)
    p0,cle2,cle1,d=fusionneAverage(df,depart)
    if(verbose):
        print("Clustering hiérarchique ascendant, version Average Linkage")
        print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
    tmp=dict(p0)
    last_key, last_value = tmp.popitem()
    somme=len(last_value)
    l=[cle2,cle1,d,somme]
    res.append(l)
    while(len(p0)>1):
        p0,cle2,cle1,d=fusionneAverage(df,p0)
        if(verbose):
            print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
        tmp=dict(p0)
        last_key, last_value = tmp.popitem()
        somme=len(last_value)
        l=[cle2,cle1,d,somme]
        res.append(l)
    if(dendrogramme):
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return res



def CHA_simple(df,verbose=False,dendrogramme=False):
    """ dendrogramme simple de df"""
    res=[]
    depart=initialise_CHA(df)
    p0,cle2,cle1,d=fusionneSimple(df,depart)
    if(verbose):
        print("Clustering hiérarchique ascendant, version Simple Linkage")
        print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
    tmp=dict(p0)
    last_key, last_value = tmp.popitem()
    somme=len(last_value)
    l=[cle2,cle1,d,somme]
    res.append(l)
    while(len(p0)>1):
        p0,cle2,cle1,d=fusionneSimple(df,p0)
        if(verbose):
            print("Distance minimale trouvée entre [",cle2,",",cle1,"] =",d)
        tmp=dict(p0)
        last_key, last_value = tmp.popitem()
        somme=len(last_value)
        l=[cle2,cle1,d,somme]
        res.append(l)
    if(dendrogramme):
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return res

        