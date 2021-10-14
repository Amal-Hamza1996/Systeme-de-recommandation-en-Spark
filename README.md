# Objectif du projet

L’objectif de ce projet est de concevoir une application Big Data et de la déployer dans une
infrastructure de calcul. Le projet se compose de deux parties, une relative à la conception d’une application de traitement de
données, l’autre à la mise en place d’une infrastructure de calcul.

# Conception d'une application Big Data

Le but de cette partie est l’implantation et l'évaluation d'une méthode de descente de gradient
pour la résolution d'un problème de fltrage collaboratif. Nous avons à notre disposition un fchier
contenant les évaluations d'un certain nombre de flms par différents utilisateurs d'une plateforme.
Ces utilisateurs n'ayant pas noté, ni même vu, l'ensemble des flms à disposition, l'objectif est de
pouvoir estimer ces notes « manquantes » depuis l'ensemble de notes, flms et utilisateurs à notre
disposition.

Les données sont stockées sous forme d'un matrice R=[rij] (ligne → utilisateur, colonne → flm,
entrée de la matrice → note), ce problème peut se modéliser comme la recherche d'une
factorisation de rang faible de R. Ceci conduit au problème d'optimisation suivant : 

![optimisation](https://user-images.githubusercontent.com/38117821/137370917-eeaed90e-6db2-4073-a90e-028ba6810d9c.PNG)


On cherche ainsi le couple de matrices P et Q, de rang fxé, tels que le produit entre P et la
transposée de Q minimise l'écart aux données R.

le fchier de données ratings.dat doit se trouver dans le répertoire /hadoop-cluster-docker/data/ 

# Mise en place d'infrastructure de calcul

Le but de cette partie est de concevoir une infrastructure Spark de calcul distribuée, simulée sur
une ordinateur personnel à l’aide de containers Docker. Un container est utilisé pour exécuter le
Master de Spark et un ensemble de containers exécutent les Slaves de Spark. Cette infrastructure
est représentée sur le schéma ci-dessous.



