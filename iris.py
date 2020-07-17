import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
from math import sqrt

'''
CARREGANDO O DATASET DENTRO DA LIBRARY SEABORN
'''
iris = sns.load_dataset("iris")
iris.head()

'''
DELETANDO A COLUNA QUE CONTEM INFORMAÇÃO SOBRE A ESPÉCIE
'''
del iris["species"];
iris.head()

plt.figure(figsize = (8,8))
sns.heatmap(iris.corr(), annot = True, linewidths = 7)
plt.show()

'''
FUNÇÃO PARA UTILIZAR O MÉTODO DO COTOVELO NO DATASET
'''
def calcula_sqic(data):
    sqic = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        sqic.append(kmeans.inertia_)

    return sqic

plt.figure(figsize=(14,5))

plt.title('ElbowMethod')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squares')
plt.grid(b = True) 
plt.xticks(range(2,21))
plt.plot(range(2,21), calcula_sqic(iris)) 
plt.plot(range(2,21), calcula_sqic(iris), '.')

y2 = calcula_sqic(iris)[len(calcula_sqic(iris))-1]
y1 = calcula_sqic(iris)[0]

plt.plot([20, 2], [y2,y1])

'''
ADICIONANDO OS PONTOS NOS GRÁFICOS
'''
for x,y in zip(range(2,21),calcula_sqic(iris)):
    label = "x{}".format(x-2)
    plt.annotate(label, (x,y), textcoords="offset points", xytext=(-5,-10), ha='right')

plt.show()

'''
FUNÇÃO PARA CALCULAR O NÚMERO IDEAL DE CLUSTERS QUE TEM COMO PARÂMETRO
UM ARRAY
'''
def numeroOtimo(a):
    x1, y1 = 2, a[0]
    x2, y2 = 20, a[len(a)-1]

    distances = []
    for i in range(len(a)):
        x0 = i+2
        y0 = a[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances))

somaDeQuadrados = calcula_sqic(iris)

'''
NÚMERO ÓTIMO DE CLUSTERS
'''

numOtimo = numeroOtimo(somaDeQuadrados)

k_medias = KMeans(n_clusters=numOtimo)
clusters = k_medias.fit_predict(iris)

'''
CONTADOR DE QUANTOS ELEMENTOS TEMOS POR CLUSTER
'''
conta1 = 0
conta2 = 0
conta0 = 0

for j in clusters:

    if j == 1:
        conta1 += 1
    elif j == 0:
        conta0 += 1
    else:
        conta2 += 1
        
print(f"O cluster 1 tem {conta1}\n O cluster 2 tem {conta2}\n O cluster 3 tem {conta0}")

'''
INSERINDO A COLUNA SPECIES NOVAMENTE NO DATASET
'''
irisAntigo = sns.load_dataset("iris")

irisAntigo['clusters'] = clusters
irisAntigo['encodedSpecies'] = preprocessing.LabelEncoder().fit_transform(irisAntigo['species'])
irisAntigo.head()

'''
PLOTANDO O COMPRIMENTE E LARGURA DA SEPALA E PETALA ANTES E DEPOIS DA CLUSTERIZAÇÃO
'''
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.title('Before')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.scatter(irisAntigo['petal_length'], irisAntigo['petal_width'], c=irisAntigo['encodedSpecies'])


plt.subplot(1, 2, 2)
plt.title('After')
plt.xlabel('petal length')
plt.scatter(irisAntigo['petal_length'], irisAntigo['petal_width'], c=irisAntigo['clusters'])

plt.show()

'''
MEDIDAS DESCRITIVAS ANTES E DEPOIS DA CLUSTERIZAÇÃO
'''
grouped_by_species = irisAntigo.groupby('species')
filtro = grouped_by_species.describe().columns.get_level_values(1).isin(['mean', 'std', 'count'])
grouped_by_species.describe().iloc[:, filtro]

irisAntigo['clusters'] = clusters
grouped_by_clusters = irisAntigo.drop('species', axis=1).groupby('clusters')

filtro = grouped_by_clusters.describe().columns.get_level_values(1).isin(['mean', 'std', 'count'])
grouped_by_clusters.describe().iloc[:, filtro]
