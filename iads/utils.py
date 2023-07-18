# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    a=np.random.uniform(binf,bsup,(n*2,p))
    b = np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])
    np.random.shuffle(b)
    return (a,b)


def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """
    Génère un ensemble de données en utilisant une distribution gaussienne.
    
    input:
        positive_center (array): Centre de la distribution gaussienne positive
        positive_sigma (array): Matrice de covariance de la distribution gaussienne positive
        negative_center (array): Centre de la distribution gaussienne négative
        negative_sigma (array): Matrice de covariance de la distribution gaussienne négative
        nb_points (int): Nombre de points à générer pour chaque classe
    
    output:
        tuple: Un tuple contenant les descriptions des données (data_desc) et les étiquettes (data_labels)
    """
    # Génération de points suivant une distribution gaussienne pour la classe positive
    points_pos=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    # Génération de points suivant une distribution gaussienne pour la classe négative
    points_neg=np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    
    data_gauss=np.concatenate((points_neg,points_pos))
    data_label=np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    
    return (data_gauss,data_label)



def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Extraction des exemples de classe -1:
    data2_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data2_positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(data2_negatifs[:,0],data2_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs[:,0],data2_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])


# ------------------------ COMPLETER LES INSTRUCTIONS DANS CETTE BOITE 
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    # Génération de points pour les quatres classes
    points1=np.random.multivariate_normal(np.array([-5,5]),np.array([[0,var],[var,0]]),n)
    points2=np.random.multivariate_normal(np.array([5,-5]),np.array([[0,var],[var,0]]),n)
    points3=np.random.multivariate_normal(np.array([-5,-5]),np.array([[0,var],[var,0]]),n)
    points4=np.random.multivariate_normal(np.array([5,5]),np.array([[0,var],[var,0]]),n)
    # Concaténation des 4 classes pour former l'ensemble description
    data_xor=np.concatenate((points1,points2,points3,points4))
    # Création des labels 
    label_xor=np.array([1]*n+[-1]*n+[1]*n+[-1]*n)
    return(data_xor,label_xor)



def colonnes_numeriques(df):
    """retourne le nom des colonnes numeriques de df
    """
    numeric_columns = df.select_dtypes(include=np.number)
    numeric_columns_np = numeric_columns.columns.tolist()
    return numeric_columns_np


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = cl.entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = cl.entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)



def partitionne(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    
    
    A_x=m_desc[:,n]
    
    first=[]
    second=[]
    
    for i in range(len(A_x)):
        if A_x[i]<=s:
            first.append(i)
        else:
            second.append(i)
            
                
    return ((m_desc[first],m_class[first]),(m_desc[second],m_class[second]))

