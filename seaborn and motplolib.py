#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Hola, Soy Diana y estoy probando un script para el curso de IA en QILAS.")


# In[3]:


(5+3)*10


# In[4]:


# guardamos la operacion en la variable "cuenta"
cuenta = (5+3)*10


# In[5]:


cuenta


# In[6]:


# guardamos la operacion en la variable "cuenta"
cuenta = (5+3)*10


# In[7]:


# accedemos al interior de la variable "cuenta" para asegurarnos que se guardo bien los datos del resultado
cuenta


# In[8]:


# el comando "type" nos permite poder saber que tipo de formato de dato esta guardado en la variable en cuestion
type(cuenta)


# In[11]:


# importamos la librería Numpy
import numpy as np


# In[12]:


# creamos la matriz de 2x2 llena de ceros con el comando np.zeros()
nula = np.zeros((2,2))


# In[13]:


# si quisieramos acceder al elemento 0,0 de nula 
nula[0,0]


# In[14]:


#Una vez declarada en python, podemos modificar alguno de los elementos de la matriz (numpy array) nula.


# por ejemplo el de la posicion 1,0
nula[1,0] = 4

# y tambien modificamos la posicion 0,1
nula[0,1] = 10


# In[18]:


# imprimimos la matriz nula en pantalla para observar la modificacion realizada
nula


# In[19]:



#podriamos realizar operaciones logicas y obtener resultados "booleanos" (True or False). Por ejemplo, preguntamos que elementos son mayores a cero.
nula > 0


# In[20]:


#Calculamos raiz cuadrada de todos los elementos de la matriz nula con el comando np.sqrt(). SQRT significa square root en ingles.

# podriamos calcular la raiz cuadrada de todos los elementos de nula con "np.sqrt()"
raiz_cuad = np.sqrt(nula)
raiz_cuad


# In[21]:


#con el comando de python de doble * se realiza el cuadrado del elemento en cuestion
# observamos que si a raiz_cuad la afectamos por el cuadrado obtenemos nuevamente nula
raiz_cuad**2


# In[22]:


#Numpy tambien nos permite crear matrices de la dimension que querramos con numeros aleatorios mediante el comando "np.random.rand()".

# creamos un array de 3x3 con todos sus elementos completados nros aleatorios de una distribución uniforme 
array_uniform = np.random.rand(3,3)
array_uniform


# In[23]:


#Podemos tambien crear vectores, solamente especificando una sola dimension.

# generamos un vector de 10 posiciones completados desde nros. aleatorios de una dist. uniforme
vector_uniform = np.random.rand(10)
vector_uniform


# In[24]:


#Trasponer un numpy array con ".T" al final
# vamos a transponer la matriz array_uniform
array_uniform_t = array_uniform.T
array_uniform_t


# In[25]:


#Calcular la inversa de una matriz (en formato numpy array) con el comando np.linalg.inv()

# podemos tambien calcularle la inversa a la matriz array_uniform
array_uniform_inv = np.linalg.inv(array_uniform)
array_uniform_inv


# In[26]:


#Calcular el producto de dos matrices con np.dot
# podemos calcular el producto punto (dot product) entre array_uniform y array_uniform_inv
array_uniform_dot = np.dot(array_uniform, array_uniform_inv)
array_uniform_dot


# In[27]:


#Crear una matriz con valores determinados por el usuario
# tambien podriamos crear una nueva matriz con datos ingresados 
a=np.array([ [1  ,-4],[12 , 3]])
a


# In[28]:


#Distribuciones de probabilidad con numpy
#Generamos un vector cuyas posiciones sean numeros aleatorios provenientes de una distribucion normal con media y desvio standard determinado.
# declaramos las variables numericas "mu" y "sigma"
mu = 0
sigma = 0.1 
# con las variables declaradas generamos un vector de 1000 posiciones
# cada posicion es un numero aleatorio extraido de una distribucion normal
normal_sample = np.random.normal(mu, sigma, 1000)


# In[29]:



# si quisieramos ver las dimensiones del vector usamos el comando np.shape
np.shape(normal_sample)


# In[30]:


# si quisieramos saber que tipo de variable es "normal_sample" usamos el comando type
type(normal_sample)


# In[31]:


# si quisieramos saber el valor de la posicion 50 del vector "normal_sample"
normal_sample[50]


# In[32]:


# podemos redondear el valor obtenido en la celda anterior con "np.round()"
# le pedimos 4 decimales
np.round(normal_sample[50],4)


# In[33]:



# imprimimos en pantalla el vector correspondiente
normal_sample


# In[48]:


#Visualizaciones con librerías Matplotlib y Seaborn
#Ambas son para realizar visualizaciones. Existen muchas formas de visualizar datos estructurados en un vector o matriz. Los mas comunes son los histogramas, los mapas de calor, etc.

# importamos las librerias que nos serviran para visualizar
import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


#Con "sns.distplot()" seaborn nos permite poder realizar un histograma y una aproximacion a la distribución de los datos en cuestion. En este caso realizaremos el distplot con el vector "normal_sample".

# generamos el distplot
sns.distplot(normal_sample)
plt.title("histograma del vector 'normal_sample' ")
plt.xlabel("valores de la variable aleatoria muestreada")
plt.ylabel("densidad de cada valor muestreado")
plt.show()


# In[50]:


# si generamos otro vector "normal_sample_2" con otra media y otro desvio standard
mu_2 = 0.5
sigma_2 = 0.2
# con las variables declaradas generamos un vector de 1000 posiciones
# cada posicion es un numero aleatorio extraido de una distribucion normal
normal_sample_2 = np.random.normal(mu_2, sigma_2, 1000)


# In[51]:


# generamos el distplot
sns.distplot(normal_sample_2)
plt.title("histograma del vector 'normal_sample_2' ")
plt.xlabel("valores de la variable aleatoria muestreada")
plt.ylabel("densidad de cada valor muestreado")
plt.show()


# In[52]:



# generamos el distplot, en una misma celda de codigo uno debajo del otro
# plt.show() imprime toda visualizacion que se haya acumulado en esa celda de codigo 
sns.distplot(normal_sample,label = "normal_sample")
sns.distplot(normal_sample_2, label = "normal_sample2")
plt.title("histograma de los dos vectores generados ")
plt.xlabel("valores de la variable aleatoria muestreada")
plt.ylabel("densidad de cada valor muestreado")
plt.legend()
plt.show()


# In[53]:


#Scatter plot Si consideramos ambos vectores como coordenadas y cada posicion una muestra (un punto) entonces tenemos 1000 muestras cada una caracterizada por 2 valores (o dimensiones). Dicho esto, podemos entonces visualizar las 1000 muestras caracterizadas por los 2 vectores provenientes cada uno de una distribucion normal con distintos parametros. El scatter plot visualiza nubes de puntos. Seaborn tiene una función para realizar scatter plots.


# In[54]:


# utilizamos cada vector como una dimension, uno para el eje x y otro para el eje y.
sns.scatterplot(x= normal_sample, y= normal_sample_2)
plt.title("Ejemplo de scatterplot con seaborn")
plt.xlabel("valores de normal_sample")
plt.ylabel("valores de normal_sample_2")
plt.show()


# In[55]:


# con el comando plt.scatter() ingresamos los 2 vectores para visualizar las 1000 muestras.
# el parametro "alpha" hace traslucidas las muestras que se solapan una encima de otra.
plt.scatter(normal_sample, normal_sample_2, alpha = 0.5)
plt.title("Ejemplo de scatterplot con matplotlib")
plt.xlabel("valores de normal_sample")
plt.ylabel("valores de normal_sample_2")
plt.show()


# In[ ]:




