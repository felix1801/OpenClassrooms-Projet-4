import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

import scipy.stats as stats
from scipy.stats import chi2_contingency, normaltest, spearmanr

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV, SelectKBest, f_regression, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

from math import sqrt

import itertools


# A quoi ressemble les valeurs des colonnes qualitatives ?
def print_column(col):
    
    missing_values = col.isnull().sum()
    total_values = len(col)
    missing_percentage = (missing_values / total_values) * 100
    print(f"Missing values: {missing_percentage}%\n")
    
    if col.dtype in ['object', 'category']:
        print(col.describe())
        print(f"\nUunique values:")
        print(col.unique())
        print('\n')

    elif col.dtype in ["float64", 'int64', 'int32']:
        # Affichage des statistiques descriptives
        print(col.describe())

        # Exclure les valeurs NaN avant de créer l'histogramme
        non_nan_values = col.dropna()

        # Création d'un histogramme avec Matplotlib
        plt.figure(figsize=(8, 6))
        plt.hist(non_nan_values, bins=30, color='blue', edgecolor='black')

        plt.xlabel(col.name)
        plt.ylabel('Fréquence')
        plt.show()

        print("\n")
        
    elif col.dtype=='bool':
        print(col.describe())
        
    else:
        print(f'Column {col.name} de type {col.dtype} non géré par la fonction print_column(col)')
		
		
		
def print_pca(cols, x, y):

	df_filled = cols.fillna(cols.mean())

	X = df_filled.values

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	n_components = df_filled.shape[1]
	pca = PCA(n_components=n_components)

	pca.fit(X_scaled)

	scree = (pca.explained_variance_ratio_*100).round(2)

	scree_cum = scree.cumsum().round()

	x_list = range(1, n_components+1)

	plt.bar(x_list, scree)
	plt.plot(x_list, scree_cum,c="red",marker='o')
	plt.xlabel("rang de l'axe d'inertie")
	plt.ylabel("pourcentage d'inertie")
	plt.title("Eboulis des valeurs propres")
	plt.show(block=False)


	fig, ax = plt.subplots(figsize=(10, 9))
	for i in range(0, pca.components_.shape[1]):
		ax.arrow(0,
				 0,  # Start the arrow at the origin
				 pca.components_[0, i],  #0 for PC1
				 pca.components_[1, i],  #1 for PC2
				 head_width=0.07,
				 head_length=0.07, 
				 width=0.02,              )

		plt.text(pca.components_[0, i] + 0.05,
				 pca.components_[1, i] + 0.05,
				 cols.columns[i])

	# affichage des lignes horizontales et verticales
	plt.plot([-1, 1], [0, 0], color='grey', ls='--')
	plt.plot([0, 0], [-1, 1], color='grey', ls='--')


	# nom des axes, avec le pourcentage d'inertie expliqué
	plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
	plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

	plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))


	an = np.linspace(0, 2 * np.pi, 100)
	plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
	plt.axis('equal')
	plt.show(block=False)

#	fig = px.scatter(pca.components_, x=0, y=1)
#	fig.show()


#	total_var = pca.explained_variance_ratio_.sum() * 100
#
#	fig = px.scatter_3d(
#		pca.components_, x=0, y=1, z=2,
#		title=f'Total Explained Variance: {total_var:.2f}%',
#		labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
#	)
#	fig.show()


def print_anova_1f(col_num, col_quali, data):

    # Exécuter l'ANOVA
    model = ols(f"{col_num} ~ {col_quali}", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    F_value = anova_table["F"][0]
    p_value = anova_table["PR(>F)"][0]
    print(anova_table)
    
    tukey = pairwise_tukeyhsd(endog=data[col_num], groups=data[col_quali], alpha=0.05)
    tukey_results = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    # Tri des niveaux de la variable qualitative en fonction de la médiane de la variable numérique
    order = data.groupby(col_quali)[col_num].median().sort_values().index

    # Afficher un graphique à boîtes
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col_quali, y=col_num, data=data, order=order)

    # Rotation des étiquettes de l'axe x pour les afficher verticalement
    plt.xticks(rotation=90)

    # Ajout des lettres du test de Tukey

    means = data.groupby(col_quali)[col_num].mean().reindex(order)
    for i, (label, mean) in enumerate(means.items()):
        # Obtenez les groupes qui ne diffèrent pas significativement de ce groupe
        not_diff = tukey_results.loc[tukey_results['group1'] == label, 'group2'][tukey_results['reject'] == False].tolist()
        not_diff += tukey_results.loc[tukey_results['group2'] == label, 'group1'][tukey_results['reject'] == False].tolist()

        # Générer une lettre pour l'annotation (par exemple, A, B, AB, etc.)
        letter = ''.join(sorted(set(label[0] for label in not_diff + [label])))
        plt.text(i, mean, letter, horizontalalignment='center', size='medium', color='black')

    plt.title(f'{col_num} X {col_quali}')
    plt.show()

    row = pd.DataFrame({
        "Var_Qualitative": [col_quali],
        "Var_Quantitative": [col_num],
        "F": [F_value],
        "p-value": [p_value],
    }, index=[0])
    
    return row
    
    
def print_corr_matrix(data, met="spearman"):
    corr_matrix = data.corr(method=met) # pearson = normal ; spearman = non-normal
    sns.heatmap(corr_matrix, annot=True, fmt='.1f', square=True, cmap='vlag')
    
def print_chi2(col1, col2):
    contingency_table = pd.crosstab(col1, col2)

    chi2, p_value, dof, _ = chi2_contingency(contingency_table)
    
    row = pd.DataFrame({
        "Var_1": [col1.name],
        "Var_2": [col2.name],
        "Chi2": [chi2],
        "p-value": [p_value],
        "ddl": [dof],
    }, index=[0])
    
    return row