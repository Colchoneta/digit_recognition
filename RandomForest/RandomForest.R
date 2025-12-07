###############################################
# PRACTICA DIGIT RECOGNITION - RANDOM FOREST
# Parte de Roberto - RF con 2000 train / 500 test
###############################################

library(randomForest)
library(caret)

# 1. Cargar datos ----------------------------

train <- read.csv("C:/Users/rober/Downloads/train.csv")
test  <- read.csv("C:/Users/rober/Downloads/test.csv")

# Aseguramos que la etiqueta es factor
train$label <- factor(train$label)

# 2. Submuestreo: 2000 train, 500 test interno ----------------

set.seed(123)

# Elegimos 2500 ejemplos aleatorios del train original
idx_total   <- sample(seq_len(nrow(train)), 2500)
data_small  <- train[idx_total, ]

# De esos 2500, cogemos 2000 para entrenar y 500 para test interno
idx_train   <- sample(seq_len(nrow(data_small)), 2000)
train_data  <- data_small[idx_train, ]
test_data   <- data_small[-idx_train, ]   # el resto (500 filas)

# Aseguramos factor en las etiquetas
train_data$label <- factor(train_data$label)
test_data$label  <- factor(test_data$label)

# 3. Normalización de píxeles [0,1] ---------------------------

# Entrenamos el preprocesado SOLO sobre los datos de entrenamiento
preproc <- preProcess(train_data[ , -1], method = "range")

# Aplicamos la transformación
x_train <- predict(preproc, train_data[ , -1])
x_test  <- predict(preproc, test_data[ , -1])

# Nos aseguramos de que x_test tiene las mismas columnas que x_train
x_test <- x_test[ , colnames(x_train)]

y_train <- train_data$label
y_test  <- test_data$label

# 4. Entrenamiento del Random Forest --------------------------

set.seed(123)
rf_model <- randomForest(
  x         = x_train,
  y         = y_train,
  ntree     = 300,  
  mtry      = 30,     
  importance = TRUE
)

print(rf_model)

# 5. Evaluación en el test interno (500 ejemplos) --------------

pred_test <- predict(rf_model, x_test)

conf_mat <- confusionMatrix(pred_test, y_test)
print(conf_mat)

accuracy <- conf_mat$overall["Accuracy"]
cat("Accuracy en el test interno (500 ejemplos):", accuracy, "\n")

# 6. Importancia de variables (píxeles) -----------------------

varImpPlot(rf_model, main = "Importancia de píxeles (Random Forest)")

# 7. Modelo final entrenado con los 100 primeros dígitos -------

train_100 <- train[1:100, ]
train_100$label <- factor(train_100$label)

# Preprocesado específico para estos 100 primeros dígitos
preproc_100 <- preProcess(train_100[ , -1], method = "range")
x_train_100 <- predict(preproc_100, train_100[ , -1])

set.seed(123)
rf_model_100 <- randomForest(
  x     = x_train_100,
  y     = train_100$label,
  ntree = 200,
  mtry  = 30
)

save(rf_model_100, file = "rf_model_100.RData")
cat("Modelo 'rf_model_100' guardado en 'rf_model_100.RData'.\n")

###############################################################

