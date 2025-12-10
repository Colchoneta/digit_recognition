# SUPPORT VECTOR MACHINE MODEL

######################################

# 1) Cargamos las librerías necesarias
# dslabs para importar la Base de Datos
# e1071 para usar SVM
install.packages("dslabs")
install.packages("e1071")
library(dslabs)
library(e1071)

######################################

# 2) Preprocesamiento de los datos

# cargamos la base de datos
mnist <- read_mnist()

X_train <- mnist$train$images
X_test <- mnist$test$images
y_train <- mnist$train$labels
y_test <- mnist$test$labels

# normalizamos los valores para que estén entre 0 y 1
X_train <- X_train / 255
X_test <- X_test / 255

#seleccionamos una muestra de los datos para train y test
#train 1000, test 250 (80/20)
set.seed(123)
train_indexes <- sample(1:nrow(X_train), 1000)
test_indexes <- sample(1:nrow(X_test), 250)

X_train_small <- X_train[train_indexes,]
X_test_small <- X_test[test_indexes,]
y_train_small <- y_train[train_indexes]
y_test_small <- y_test[test_indexes] 

######################################

# 3) Búsqueda de hiperparámetros

#buscamos hiperparámetros
tuned <- e1071::tune(
  METHOD = svm,
  train.x = X_train_small,
  train.y = as.factor(y_train_small),
  ranges = list(
    cost = c(0.1, 1, 10),
    gamma = c(0.001, 0.01, 0.1)
  )
)

#encontramos los mejores hiperparámetros
print(tuned$best.parameters)
print(tuned$best.performance)


######################################

# 4) Entrenamiento de modelo final

svm_model <- svm(
  x = X_train_small,
  y = as.factor(y_train_small),
  kernel = "radial",
  cost  = tuned$best.parameters$cost,
  gamma = tuned$best.parameters$gamma
)

summary(svm_model)

######################################

# 5) Evaluación modelo final

#predicción del modelo
pred <- predict(svm_model, X_test_small)

#matriz de confusión
matrix <- table(
  Predicted = pred,
  Actual = y_test_small
  )
print(matrix)

#calculamos accuracy
accuracy <- sum(diag(matrix)) / sum(matrix)
cat("Precisión del modelo SVM:", accuracy, "\n")

######################################

# 6) Guardar modelo final

saveRDS(svm_model, "svm_model.RData")
cat("\nModelo guardado como svm_model.RData\n")