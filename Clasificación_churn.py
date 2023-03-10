#!/usr/bin/env python
# coding: utf-8

# # Estudio comparativo de clasificadores para la predicción de fuga de clientes en la empresa Orange Telecom

# In[114]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Primero se importan los datos del conjunto de Prueba

# In[115]:


df1=pd.read_csv("churn-bigml-20.csv", sep=",") 
df1


# Y ahora los datos del conjunto de Entrenamiento

# In[116]:


df2=pd.read_csv("churn-bigml-80.csv", sep=",")
df2


# Primero se concatenarán las dos bases de datos, debido que no necesitaremos tenerlas por separado.

# In[117]:


df= pd.concat([df1,df2],axis=0)
df.index=pd.Series(range(0,3333)) #se arregla el índice para que valla del 0 al 3332, con el concat se concatenaron los índices
df.head()


# Se obtienen estadísticas básicas, tales como la media, desviación estándar, mínimo, máximo y algunos cuartiles.

# In[118]:


df.describe()


# ¿Hay datos perdidos? con dropna(axis=0) se eliminan las filas que contengan datos perdidos, por tanto si no se elimina ninguna se podría concluir que no existen valores NaN.

# In[119]:


df.dropna(axis = 0)


# Se logra apreciar que no se eliminó ninguna fila, por tanto, se concluye que no hay valores NaN.

# Se obtiene un gráfico de torta de la variable de respuesta churn.

# In[ ]:


churn=df.groupby((df['Churn'])).contador.sum() #se va sumando el contador en cada Estado según frecuencia
churn.head()


# In[ ]:


colores = ["#60D394","#FF9B85"]

plt.pie(churn,labels=["Sigue en la empresa"," Abandonó la empresa"], autopct="%0.1f %%",colors=colores)
plt.title('¿El usuario sigue en la empresa?', fontsize=20)
plt.axis("equal")
plt.show()


# En consiguiente, un gráfico de barras sobre la cantidad de usuarios por cada estado:

# In[ ]:


estados=df.groupby((df['State'])).contador.sum()

plt.figure(figsize=(40,10))
plt.title("N° de usuarios inscritos en la empresa según su State",fontsize=40)
plt.xlabel('State', fontsize=25)
plt.ylabel('N° de usuarios', fontsize=25)
plt.xticks(rotation=45,ha="right")
estados.plot.bar()


# Otro gráfico:

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(df["Customer service calls"],hue = df["Churn"],palette = "dark")
plt.title("¿El usuario sigue en la empresa después de realizar llamadas a servicio al cliente?",fontsize=16)
plt.xlabel('Total de llamadas a servicio al cliente', fontsize = 12)
plt.ylabel('N° de usuarios', fontsize = 12)


# Ahora se verán boxplot sobre el total de minutos en las categorías de day, eve, night y intl, para poder comparar el gasto de minutos en las diferentes variables.

# In[120]:


data1 =df["Total day minutes"]
data2 =df["Total eve minutes"]
data3 =df["Total night minutes"]
data4 =df["Total intl minutes"]

databox1=[data1,data2,data3,data4]
  
fig = plt.figure(figsize =(10, 7)) 
  
ax = fig.add_axes([0, 0, 1, 1]) 
  
bp = ax.boxplot(databox1) 
  
plt.show() 


# se logra apreciar que en todas las categorías del gasto de minutos hay personas que gastan cifras que se escapa del promedio de los clientes, además se puede apreciar que el comportamiento de gasto en las 4 variables se comporta de manera simétrica y la categoría en donde la mediana es más alta es en la de total eve minutes, seguido por el total day minutes. 
# 
# Cabe destacar, que la variable total intl minutes es muy pequeño el gasto de minutos, esto debe ser debido a que hay que pagar más para poder tener el plan internacional, por lo que es más caro.
# 
# Ahora se realiza lo mismo pero sobre el total de llamadas en las categorías de day, eve, night y intl, para poder comparar el total de llamadas en las diferentes variables.

# In[121]:


data5 =df["Total day calls"]
data6 =df["Total eve calls"]
data7 =df["Total night calls"]
data8 =df["Total intl calls"]

databox2=[data5,data6,data7,data8]
  
fig = plt.figure(figsize =(10, 7)) 
  
ax = fig.add_axes([0, 0, 1, 1]) 
  
bp = ax.boxplot(databox2) 
  
plt.show() 


# Nuevamente la categoría intl resulta tener valores demasiado menores respecto a las demás, siguen habiendo valores atípicos en esta sección y la simetría de las distribuciones se mantiene. También ahora se iguala la mediana en las 3 primeras categorías, lo que indica una mayor homogeneidad en sus valores.
# 
# Por último, se hará el mismo gráfico pero ahora para la categoría del total de recarga:

# In[122]:


data9 =df["Total day charge"]
data10 =df["Total eve charge"]
data11 =df["Total night charge"]
data12 =df["Total intl charge"]

databox3=[data9,data10,data11,data12]
  
fig = plt.figure(figsize =(10, 7)) 
  
ax = fig.add_axes([0, 0, 1, 1]) 
  
bp = ax.boxplot(databox3) 
  
plt.show() 


# Violin plot del total de minutos en la noche según el uso del plan internacional

# In[ ]:


#Violinplot
fig7, ax7 = plt.subplots()
ax7.violinplot(df_TNMxIPsi, vert = False)
plt.figure(figsize=(24,6))
ax7.set_title('Total de minutos en la noche según el uso de plan internacional')
ax7.set_xlabel('Total de minutos usados en la noche', fontsize = 12)
ax7.set_ylabel('¿Usa plan internacional?', fontsize = 12)

ax7.violinplot(df_TNMxIPno, vert = False)
plt.figure(figsize=(24,6))
ax7.legend()
plt.show()


# Para finalizar, se hará un análisis de correlación entre todas las variables para poder identificar si hay indicios de existencia de multicolinealidad, lo que puede afectar a estudios posteriores para la predicción de datos (por ejemplo, puede afectar a la regresión logística).

# In[123]:


corr_df = df.corr(method='pearson')
corr_df


# In[124]:


plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()


# Se logra apreciar que en general las variables no están altamente correlacionadas, solamente las variables total charge y total minutes (en sus 4 diferentes categorías), lo que es lógico. Se espera que posteriormente con el standarscaler se logre apalear este fenómeno. Por el momento se observa que en general no hay multicolinealidad y solo es por estas variables que hay que tener cuidado.

# Para reafirmar lo anterior, se hace un gráfico pairplot a través de la librería seaborn:

# In[125]:


sns.pairplot(df)


# Se aprecia rotundamente que la correlación entre la variable charge y los minutos que ocupa una persona (ya sean en el día, noche, etc.) están correlacionados linealmente con un valor de apróx 1. El resto de variables no se observa que contengan una correlación entre si.

# Se separa el conjunto en entrenamiento y prueba. Primero, se seleccionan las variables que se ocuparán como X (no se toma en cuenta los estados ya que son variables categóricas (y apróx 50 estados diferentes) por tanto no aportarán a los clasificadores, además de no poder usar el standard scaler en ellos.

# In[126]:


diccionariocategorico={"Yes":1,"No":0}  #se cambian los yes y no por 0
df=df.replace(diccionariocategorico)

x= df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]  ##Se selecciona las variables que ocuparemos como X
x.head()


# Ahora se selecciona la variable Churn que se ocupará como Y.

# In[127]:


diccionariobooleano = {True: 1, False: 0} #para cambiar los True y False, ya que se consideran booleanos
df = df.replace(diccionariobooleano)

y=df.iloc[:,[19]]
y.head()


# Ahora se separa el conjunto de datos en entrenamiento(70%) y test (30%):

# In[128]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, 
                test_size=0.3, random_state=1, stratify=y)


# In[129]:


X_train.shape


# In[130]:


X_test.shape


# In[131]:


y_train.shape


# In[132]:


y_test.shape


# #### 7. Estandarizar los atributos utilizando el StandardScaler
# 
# Con el fin de que todos los atributos tenga igual importancia al entrenar un clasificador, y debido a que los clasificadores están optimizados para los datos escalados. Procederemos a reescalar los datos.
# 
# En particular, el escalamiento que se aplicará es centrar los datos en 0 y cambiar su variabilidad a 1:
# 
# $$Z=\frac{X - \overline{X}}{S_{X}}$$
# 
# donde $\overline{X}$ corresponde al promedio de los datos y $S_{X}$ a la desviación estándar de los datos de entrada.
# 
# Con la función StandardScaler() de la librería sklearn.preprocessing se puede hacer lo anterior:

# In[133]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
Z_train = sc.transform(X_train) #tanto el conjunto de entrenamiento como el de test son reescalados
Z_test = sc.transform(X_test)


# LDA (Linear Discriminant Analysis)

# In[134]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
ldamodel = lda.fit(Z_train, y_train)

predldamodel=ldamodel.predict(Z_test)

confusion_matrix(y_test, predldamodel)


# QDA (Quadratic Discriminant Analysis)

# In[135]:


qda = QuadraticDiscriminantAnalysis()
qdamodel = qda.fit(Z_train, y_train)

predqdamodel=qdamodel.predict(Z_test)

confusion_matrix(y_test, predqdamodel)


# Árbol de decisión

# In[136]:


from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier(criterion = 'gini', max_depth=4)
arbolmodel=arbol.fit(Z_train, y_train)

predarbolmodel=arbolmodel.predict(Z_test)

confusion_matrix(y_test,predarbolmodel)


# Random Forest

# In[137]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(criterion = 'gini', 
                                 n_estimators=25, n_jobs=2)
randomforestmodel=randomforest.fit(Z_train, y_train)

predrandomforestmodel=randomforestmodel.predict(Z_test)
confusion_matrix(y_test,predrandomforestmodel)


# Regresión Logística

# In[138]:


from sklearn.linear_model import LogisticRegression

reglogis = LogisticRegression(C=100.0)
reglogismodel=reglogis.fit(Z_train, y_train)

predreglogismodel=reglogismodel.predict(Z_test)

confusion_matrix(y_test,predreglogismodel)


# SVC lineal

# In[139]:


from sklearn.svm import SVC

svclineal = SVC(C = 100, kernel = 'linear')
svclinealmodel=svclineal.fit(Z_train, y_train)

predsvclinealmodel=svclinealmodel.predict(Z_test)

confusion_matrix(y_test,predsvclinealmodel)                       


# SVC radio basal

# In[140]:


svcrb=SVC(C=100, kernel='rbf')
svcrbmodel=svcrb.fit(Z_train,y_train)

predsvcrbmodel=svcrbmodel.predict(Z_test)

confusion_matrix(y_test,predsvcrbmodel)


# Perceptron

# In[141]:


from sklearn.linear_model import Perceptron

perceptron = Perceptron(max_iter=40, eta0=0.1)
perceptronmodel=perceptron.fit(Z_train, y_train)

predperceptronmodel=perceptronmodel.predict(Z_test)

confusion_matrix(y_test,predperceptronmodel)


# Perceptrón Multicapa

# In[142]:


from sklearn.neural_network import MLPClassifier

perceptronmc = MLPClassifier(random_state=1, max_iter=300)
perceptronmcmodel = perceptronmc.fit(Z_train, y_train)

predperceptronmcmodel=perceptronmcmodel.predict(Z_test)

confusion_matrix(y_test,predperceptronmcmodel)


# K-Neighbors Classifier

# In[143]:


from sklearn.neighbors import KNeighborsClassifier

kneighbor = KNeighborsClassifier(n_neighbors=5)
kneighbormodel=kneighbor.fit(Z_train, y_train)

predkneighbormodel=kneighbormodel.predict(Z_test)

confusion_matrix(y_test,predkneighbormodel)


# Gradient Boosting

# In[144]:


from sklearn.ensemble import GradientBoostingClassifier

gradientboos = GradientBoostingClassifier()

gradientboosmodel=gradientboos.fit(Z_train, y_train)

predgradientboosmodel=gradientboosmodel.predict(Z_test)

confusion_matrix(y_test,predgradientboosmodel)


# XGBoost

# In[145]:


import xgboost as xgb

xgboost = xgb.XGBClassifier()
xgboostmodel=xgboost.fit(Z_train, y_train)

predxgboostmodel=xgboostmodel.predict(Z_test)

confusion_matrix(y_test,predxgboostmodel)


# #### 9. Obtener las métricas de desempeño para todos los clasificadores (Accuracy, Recall, Precision, Specificity, F-measure) y construir una tabla comparativa.

# In[146]:


from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


# In[147]:


##Accuracy
acc1=accuracy_score(y_test, predldamodel)
acc2=accuracy_score(y_test, predqdamodel)
acc3=accuracy_score(y_test, predarbolmodel)
acc4=accuracy_score(y_test, predrandomforestmodel)
acc5=accuracy_score(y_test, predreglogismodel)
acc6=accuracy_score(y_test, predsvclinealmodel)
acc7=accuracy_score(y_test, predsvcrbmodel)
acc8=accuracy_score(y_test, predperceptronmodel)
acc9=accuracy_score(y_test, predperceptronmcmodel)
acc10=accuracy_score(y_test,predkneighbormodel)
acc11=accuracy_score(y_test,predgradientboosmodel)
acc12=accuracy_score(y_test,predxgboostmodel)


# In[148]:


#Recall
recall1=recall_score(y_test, predldamodel)
recall2=recall_score(y_test, predqdamodel)
recall3=recall_score(y_test, predarbolmodel)
recall4=recall_score(y_test, predrandomforestmodel)
recall5=recall_score(y_test, predreglogismodel)
recall6=recall_score(y_test, predsvclinealmodel)
recall7=recall_score(y_test, predsvcrbmodel)
recall8=recall_score(y_test, predperceptronmodel)
recall9=recall_score(y_test, predperceptronmcmodel)
recall10=recall_score(y_test,predkneighbormodel)
recall11=recall_score(y_test,predgradientboosmodel)
recall12=recall_score(y_test,predxgboostmodel)


# In[149]:


## Precisión
prec1=precision_score(y_test, predldamodel)
prec2=precision_score(y_test, predqdamodel)
prec3=precision_score(y_test, predarbolmodel)
prec4=precision_score(y_test, predrandomforestmodel)
prec5=precision_score(y_test, predreglogismodel)
prec6=precision_score(y_test, predsvclinealmodel)
prec7=precision_score(y_test, predsvcrbmodel)
prec8=precision_score(y_test, predperceptronmodel)
prec9=precision_score(y_test, predperceptronmcmodel)
prec10=precision_score(y_test,predkneighbormodel)
prec11=precision_score(y_test,predgradientboosmodel)
prec12=precision_score(y_test,predxgboostmodel)


# No se encontró una función de sklearn.metrics para calcular specificity, por lo tanto se hará manualmente mediante la matriz de confusión.
# Recordar que:
# Specificity=$\frac{VN}{FP+VN}$
# 
# Y además, la función confusion_matrix de la librería sklearn entrega la matriz con los resultados al revés, por tanto, se deberá hacer los cálculos manuales al revés también. ( https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html ) 

# In[163]:



cm1=confusion_matrix(y_test, predldamodel)
cm2=confusion_matrix(y_test, predqdamodel)
cm1=confusion_matrix(y_test, predarbolmodel)
cm1=confusion_matrix(y_test, predrandomforestmodel)
cm5=confusion_matrix(y_test, predreglogismodel)
cm6=confusion_matrix(y_test, predsvclinealmodel)
cm7=confusion_matrix(y_test, predsvcrbmodel)
cm8=confusion_matrix(y_test, predperceptronmodel)
cm9=confusion_matrix(y_test, predperceptronmcmodel)
cm10=confusion_matrix(y_test, predkneighbormodel)
cm11=confusion_matrix(y_test,predgradientboosmodel)
cm12=confusion_matrix(y_test,predxgboostmodel)

#specificity
specificity1 = cm1[0,0]/ (cm1[0,0]+cm1[0,1])
specificity2 = cm2[0,0]/ (cm2[0,0]+cm2[0,1])
specificity3 = cm3[0,0]/ (cm3[0,0]+cm3[0,1])
specificity4 = cm4[0,0]/ (cm4[0,0]+cm4[0,1])
specificity5 = cm5[0,0]/ (cm5[0,0]+cm5[0,1])
specificity6 = cm6[0,0]/ (cm6[0,0]+cm6[0,1])
specificity7 = cm7[0,0]/ (cm7[0,0]+cm7[0,1])
specificity8 = cm8[0,0]/ (cm8[0,0]+cm8[0,1])
specificity9 = cm9[0,0]/ (cm9[0,0]+cm9[0,1])
specificity10 = cm10[0,0]/ (cm10[0,0]+cm10[0,1])
specificity11 = cm11[0,0]/ (cm11[0,0]+cm11[0,1])
specificity12 = cm12[0,0]/ (cm12[0,0]+cm12[0,1])


# In[164]:


#medida F
f1=f1_score(y_test, predldamodel)
f2=f1_score(y_test, predqdamodel)
f3=f1_score(y_test, predarbolmodel)
f4=f1_score(y_test, predrandomforestmodel)
f5=f1_score(y_test, predreglogismodel)
f6=f1_score(y_test, predsvclinealmodel)
f7=f1_score(y_test, predsvcrbmodel)
f8=f1_score(y_test, predperceptronmodel)
f9=f1_score(y_test, predperceptronmcmodel)
f10=f1_score(y_test,predkneighbormodel)
f11=f1_score(y_test,predgradientboosmodel)
f12=f1_score(y_test,predxgboostmodel)


# In[165]:


# TABLA COMPARATIVA:

clasificador=["LDA","QDA", "Arbol de decisión","Random Forest", "Regresión Logís","SVC Lineal","SVC Radio basal",
              "Perceptrón","Perceptrón Multicapa","K-neighbors","Gradient Boosting","XGBoost"]

accuracy=[acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,acc10,acc11,acc12]

recall=[recall1,recall2,recall3,recall4,recall5,recall6,recall7,recall8,recall9,recall10,recall11,recall12]

precision=[prec1,prec2,prec3,prec4,prec5,prec6,prec7,prec8,prec9,prec10,prec11,prec12]

specificity=[specificity1,specificity2,specificity3,specificity4,specificity5,specificity6,specificity7,specificity8,
            specificity9,specificity10,specificity11,specificity12]

fscore=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]

pd.DataFrame({"Clasificador": clasificador,"Accuracy":accuracy,"Recall":recall, "Precisión":precision,
              "Especificidad":specificity, "Medida F":fscore
             })

