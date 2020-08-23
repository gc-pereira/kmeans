from pylab import rand,plot,show,norm
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import scipy.optimize as opt  
import os  

''' Definição das funções '''
# Define a função sigmoide
def sigmoid(z):  
    return 1 / (1 + np.exp(­z))

# Plota curva para visualização
def plot_sigmoid():
    nums = np.arange(­10, 10, step=0.5)
    fig, ax = plt.subplots(figsize=(9,5))  
    ax.plot(nums, sigmoid(nums), 'r')  

# Define o custo para os parâmetros atuais 
# É o negativo da log­verossimilhança
def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(­y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 ­ y), np.log(1 ­ sigmoid(X * theta.T)))
    
    return np.sum(first ­ second) / (len(X))

# Calcula o gradiente no ponto
# Usado na descida do gradiente
def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) ­ y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad

# Realiza a predição da classe (0 ou 1) para os parâmetros encontrados
def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

''' Início do script '''
path = os.getcwd() + '/ex2data1.txt'  
data = pd.read_csv(path,header=None,names=['Exam1','Exam2','Admitted'])
print(data.head())

# Plota a sigmoide para visualização
plot_sigmoid()

# Testa função de custo
# Adiciona aos dados uma coluna de 1's
data.insert(0, 'Ones', 1)

# X  são os dados de treinamento e y  são os rótulos
cols = data.shape[1]  
X = data.iloc[:,0:cols­1]  
y = data.iloc[:,cols­1:cols]

# converte para vetores do numpy e inicializa vetor de pesos com 0's
X = np.array(X.values)  
y = np.array(y.values)  
theta = np.zeros(3)

# encontra os valores de theta ótimos, ou seja, que minimizam 
# função de custo (custo = negativo da verossimilhança)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

# result[0] é o vetor de parâmetros estimado por máxima verossimilhança
print('Parâmetros ótimos: ', result[0])
print('Custo mínimo: ', cost(result[0], X, y))

# calcula a acurária
theta_min = np.matrix(result[0])  
predictions = predict(theta_min, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for
(a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print('Acurácia = {0}%'.format(accuracy))

# Plota superfície de separação e os dados
positive = data[data['Admitted'].isin([1])]  
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(9,5))  
ax.scatter(positive['Exam   1'],   positive['Exam   2'],   s=50,   c='b',
marker='o', label='Admitted')  
ax.scatter(negative['Exam   1'],   negative['Exam   2'],   s=50,   c='r',
marker='x', label='Not Admitted')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')
theta = result[0]
x = np.linspace(30, 100, 100)
y = ­(theta[0] + theta[1]*x)/theta[2]
plt.plot(x, y, 'k­­')
