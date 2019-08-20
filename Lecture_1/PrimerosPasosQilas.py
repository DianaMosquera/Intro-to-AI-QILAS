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


# In[ ]:




