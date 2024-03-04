# Machine learning for finance

# Système de Trading Algorithmique  

Ce projet Python (MLTradingStrategy.py) démontre comment construire un système de trading algorithmique basique qui utilise l'analyse technique et l'apprentissage automatique pour prendre des décisions de trading. Le système télécharge les données historiques des prix d'actions, calcule divers indicateurs techniques, utilise un modèle de machine learning pour prédire la direction future des prix, et simule des décisions de trading basées sur ces prédictions.

## Fonctionnalités

- Téléchargement des données historiques des prix d'actions via `yfinance`.
- Calcul d'indicateurs techniques tels que la moyenne mobile exponentielle (EMA), l'indice de force relative (RSI), et le MACD.
- Utilisation d'un modèle de forêt aléatoire pour prédire la direction future des prix.
- Simulation de stratégie de trading basée sur les prédictions du modèle.
- Visualisation des retours de la stratégie par rapport aux retours du marché.

## Prérequis

Assurez-vous d'avoir Python 3.x installé sur votre système. Vous aurez également besoin d'installer les bibliothèques suivantes :

- numpy
- pandas
- scikit-learn
- matplotlib
- yfinance

Vous pouvez installer toutes les dépendances nécessaires en exécutant :

```bash
pip install numpy pandas scikit-learn matplotlib yfinance
```


## Utilisation

Pour exécuter le script, suivez ces étapes :

1. Ouvrez votre terminal ou invite de commande.
2. Naviguez jusqu'au répertoire contenant le script `TradingSystem.py`.
3. Exécutez le script en tapant :

```bash
python TradingSystem.py
```


## Personnalisation

Vous pouvez personnaliser le script en modifiant :

- La liste des symboles boursiers à suivre en ajustant la variable `stocks`.
- La période des données historiques en modifiant les paramètres `start` et `end` dans la fonction `yf.download`.
- Les indicateurs techniques calculés et utilisés pour la prédiction.
- Le modèle de machine learning en remplaçant le modèle de forêt aléatoire par un autre modèle de votre choix.

## Contribution

Les contributions à ce projet sont les bienvenues. Vous pouvez contribuer en :

- Améliorant l'efficacité du code.
- Ajoutant de nouveaux indicateurs techniques.
- Expérimentant avec différents modèles de machine learning pour améliorer la précision des prédictions.
- Améliorant la stratégie de trading pour augmenter les retours sur investissement.


