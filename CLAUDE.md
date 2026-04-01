# Robustesse OOD — Contexte du projet

## Objectif
Librairie de **certification de détection OOD** de réseaux de neurones profonds (DNNs, ResNets, ...).

L'objectif est de vérifier post-hoc qu'un input est OOD à partir des bornes obtenues par la librairie alpha-beta-CROWN (qui calcule des bornes de vecteurs pré-activations quand on considère une perturbation normée autour d'un input. Il s'agit de bornes calculées dans le contexte de robustesse adversariale).

Le point clé du projet : trouver un critère de répartition entre données OOD et ID. Les différentes idées de  critère sont : 
- Proportion de neurones stables actifs
- Proportion de neurones stables inactifs
- Gap trouvé entre borne supérieure et bornes inférieures obtenues avec alpha-beta-CROWN.
- etc
---

## Premiers pas à faire
- Statistiques sur les proportion de neurones stables actifs, stables inactifs, gaps, etc sur les données OOD et données ID.