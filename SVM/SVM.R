# SUPPORT VECTOR MACHINE MODEL

######################################

# 1) Cargamos las librerías necesarias
# keras para importar la Base de Datos
# e1071 para usar SVM
install.packages("keras")
library(keras)
install.packages("e1071")
library(e1071)

######################################

# 2) Preprocesamiento de los datos

# cargamos la base de datos
mnist <- dataset_mnist()
X_train <- mnist$train$x
X_test <- mnist$test$x
y_train <- mnist$train$y
y_test <- mnist$test$y

# normalizamos los valores para que estén entre 0 y 1
# transformamos cada patrón x de una matriz de 28x28 a un vector de 784
X_train <- array_reshape(X_train, c(nrow(X_train), 784)) / 255
X_test <- array_reshape(X_test, c(nrow(X_test), 784)) / 255

#seleccionamos una muestra de los datos para train y test
#train 2000, test 500 (80/20)
set.seed(123)
train_indexes <- sample(1:nrow(X_train), 1000)
test_indexes <- sample(1:nrow(X_test), 250)

X_train_small <- X_train[train_indexes,]
X_test_small <- X_test[test_indexes,]
y_train_small <- y_train[train_indexes] 
y_test_small <- y_test[test_indexes]




