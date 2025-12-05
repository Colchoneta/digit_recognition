# ==============================================================================
# 1. CARGA DE PAQUETES
# ==============================================================================
library(readr)      
library(dplyr)       
library(keras3)      
library(caret)        
set.seed(123)       

# ==============================================================================
# 2. CARGA Y PREPROCESAMIENTO DE LOS DATOS
# ==============================================================================
mnist <- dataset_mnist() 

# Selección de las primeras 1000 imágenes de entrenamiento y 250 de prueba
x_train <- mnist$train$x[1:1000, , ]  
y_train <- mnist$train$y[1:1000]      
x_test <- mnist$test$x[1:250, , ]     
y_test <- mnist$test$y[1:250]         

# Redimensionamiento de las imágenes: 28x28 se convierte en un vector de 784 características y normalización
x_train <- array_reshape(x_train, c(nrow(x_train), 784)) / 255  
x_test <- array_reshape(x_test, c(nrow(x_test), 784)) / 255    

# Codificación de las etiquetas en formato one-hot (vectores binarios)
y_train <- to_categorical(y_train, num_classes = 10)  
y_test <- to_categorical(y_test, num_classes = 10)

# ==============================================================================
# 3. ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# ==============================================================================
# Aplica el PCA a los datos de entrenamiento
pca <- prcomp(x_train, center = TRUE, scale. = FALSE)  


# Calcula la varianza explicada por cada componente principal
var_explicada <- pca$sdev^2 / sum(pca$sdev^2)  
var_acumulada <- cumsum(var_explicada)          

# Determina el número mínimo de componentes necesarios para explicar el 95% de la varianza
n_comp_95 <- which(var_acumulada >= 0.95)[1]   

# Reducción de dimensionalidad: selecciona solo los primeros 'n_comp_95' componentes principales
x_train <- as.matrix(pca$x[, 1:n_comp_95]) 
x_test  <- predict(pca, x_test)[, 1:n_comp_95]