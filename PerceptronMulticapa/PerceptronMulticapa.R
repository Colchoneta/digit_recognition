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
# 3. GRID SEARCH (Layers x Dropout Rate x Unidades Iniciais)
# ==============================================================================

# Hiperparâmetros a testar
layers_to_test <- c(1, 2, 3)          # Número de Camadas Ocultas
initial_units_to_test <- c(256, 128)   # Unidades para a primeira camada
dropout_rates_to_test <- c(0.2, 0.4)  # Taxas de Dropout

results_full_grid <- list()
counter <- 1

for (L in layers_to_test) {
  for (units in initial_units_to_test) {
    for (rate in dropout_rates_to_test) {
      cat(paste("--- Testando L=", L, " | Unidades:", units, " | Dropout:", rate, " ---\n", sep=""))
      
      # 1. Definir o Modelo
      model_grid_test <- keras_model_sequential()
      current_units <- units
      
      # Construir dinamicamente L camadas ocultas
      for (i in 1:L) {
        # Adicionar camada densa
        model_grid_test <- model_grid_test %>%
          layer_dense(units = current_units, activation = 'relu',
                      input_shape = if (i == 1) n_comp_95 else NULL)
        
        # Adicionar camada de dropout (aplicada após cada camada oculta)
        model_grid_test <- model_grid_test %>%
          layer_dropout(rate = rate)
        
        # Reduzir unidades para a próxima camada (metade, com mínimo de 32)
        current_units <- max(32, floor(current_units / 2)) 
      }
      
      # Adicionar a camada de saída
      model_grid_test <- model_grid_test %>%
        layer_dense(units = 10, activation = 'softmax')
      
      # 2. Compilar o Modelo
      model_grid_test %>% compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = c('accuracy')
      )
      
      # 3. Treinar o Modelo
      model_grid_test %>% fit(
        x_train, y_train,
        epochs = 10,
        batch_size = 128,
        validation_split = 0.2,
        verbose = 0
      )
      
      # 4. Avaliar o Modelo
      score <- model_grid_test %>% evaluate(x_test, y_test, verbose = 0)
      
      # 5. Armazenar o Resultado
      results_full_grid[[counter]] <- list(
        Layers = L,
        Initial_Units = units,
        Dropout_Rate = rate,
        Loss = score$loss,
        Accuracy = score$accuracy
      )
      
      cat(paste("-> Accuracy:", round(score$accuracy, 4), "\n"))
      counter <- counter + 1
    }
  }
}

# ==============================================================================
# 4. ANÁLISE DOS MELHORES HIPERPARAMETROS
# ==============================================================================
results_df_full_grid <- do.call(rbind, lapply(results_full_grid, as.data.frame))

# Ordenar por Accuracy (o melhor no topo)
results_df_full_grid <- results_df_full_grid[order(-results_df_full_grid$Accuracy), ]

best_params <- results_df_full_grid[1, ]

L <- best_params$Layers
unidades_iniciales <- best_params$Initial_Units
dropout_rate <- best_params$Dropout_Rate

