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


# ==============================================================================
# 3. BÚSQUEDA EN MALLA (GRID SEARCH) (Capas x Tasa de Dropout x Unidades Iniciales)
# ==============================================================================

# Hiperparámetros a probar
layers_to_test <- c(1, 2, 3)           # Número de Capas Ocultas
initial_units_to_test <- c(256, 128)   # Unidades para la primera capa
dropout_rates_to_test <- c(0.2, 0.4)   # Tasas de Dropout

results_full_grid <- list()
counter <- 1

for (L in layers_to_test) {
  for (units in initial_units_to_test) {
    for (rate in dropout_rates_to_test) {
      # 1. Definir el Modelo
      model_grid_test <- keras_model_sequential()
      current_units <- units
      
      # Construir dinámicamente L capas ocultas
      for (i in 1:L) {
        # Añadir capa densa
        model_grid_test <- model_grid_test %>%
          layer_dense(units = current_units, activation = 'relu',
                      input_shape = if (i == 1) n_comp_95 else NULL)
        
        # Añadir capa de dropout (aplicada después de cada capa oculta)
        model_grid_test <- model_grid_test %>%
          layer_dropout(rate = rate)
        
        # Reducir unidades para la próxima capa (mitad, con mínimo de 32)
        current_units <- max(32, floor(current_units / 2)) 
      }
      
      # Añadir la capa de salida
      model_grid_test <- model_grid_test %>%
        layer_dense(units = 10, activation = 'softmax')
      
      # 2. Compilar el Modelo
      model_grid_test %>% compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = c('accuracy')
      )
      
      # 3. Entrenar el Modelo
      model_grid_test %>% fit(
        x_train, y_train,
        epochs = 10,
        batch_size = 128,
        validation_split = 0.2,
        verbose = 0
      )
      
      # 4. Evaluar el Modelo
      score <- model_grid_test %>% evaluate(x_test, y_test, verbose = 0)
      
      # 5. Almacenar el Resultado
      results_full_grid[[counter]] <- list(
        Layers = L,
        Initial_Units = units,
        Dropout_Rate = rate,
        Loss = score$loss,
        Accuracy = score$accuracy
      )
      counter <- counter + 1
    }
  }
}

# ==============================================================================
# 4. ANÁLISIS DE LOS MEJORES HIPERPARÁMETROS
# ==============================================================================
# Combina los resultados de todos los modelos probados en el Grid Search en un único data frame
results_df_full_grid <- do.call(rbind, lapply(results_full_grid, as.data.frame))

# Ordena el data frame por Accuracy (en orden descendente, el mejor modelo queda primero)
results_df_full_grid <- results_df_full_grid[order(-results_df_full_grid$Accuracy), ]

# Selecciona los parámetros del modelo con la mayor Accuracy (la primera fila)
best_params <- results_df_full_grid[1, ]

# Asigna los valores de los mejores hiperparámetros a variables
L <- best_params$Layers
unidades_iniciales <- best_params$Initial_Units
dropout_rate <- best_params$Dropout_Rate

# Mostrar los resultados del mejor modelo
cat(paste("Accuracy en el test interno:", round(best_params$Accuracy, 4), "\n"))
cat(paste("Loss en el test interno:", round(best_params$Loss, 4), "\n"))
cat("\n")

# ==============================================================================
# 5. CREACION DEL MODELO PARA LOS 100 PRIMEROS DATOS
# ==============================================================================

# Selección de los primeros 100 datos para el entrenamiento final
x_train_100 <- x_train[1:100, ]
y_train_100 <- y_train[1:100, ]

# 1. Definir el Modelo Final (estructura basada en best_params)
final_model <- keras_model_sequential()
current_units <- unidades_iniciales

# Construir las L capas ocultas
for (i in 1:L) {
  final_model <- final_model %>%
    layer_dense(units = current_units, activation = 'relu',
                input_shape = if (i == 1) n_comp_95 else NULL)
  
  final_model <- final_model %>%
    layer_dropout(rate = dropout_rate)
  
  current_units <- max(32, floor(current_units / 2)) 
}

# Capa de salida
final_model <- final_model %>%
  layer_dense(units = 10, activation = 'softmax')

# 2. Compilar el Modelo Final
final_model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# 3. Entrenar el Modelo Final 
history_final <- final_model %>% fit(
  x_train_100, y_train_100, 
  epochs = 0, 
  batch_size = 32, 
  verbose = 1 
)
save(final_model, file = "perceptron_model_100.RData")
cat("Modelo guardado en 'percetron_model_100.RData'.\n")

