library(rpart)
library(rpart.plot)
library(caret)
library(dslabs)
library(adabag)

########################
##  CREACION DATASET  ##
########################

mnist <- read_mnist()
set.seed(123)
idx <- sample(nrow(mnist$train$images), 1250)
mnist$train$images <- mnist$train$images[idx, ]
mnist$train$labels <- mnist$train$labels[idx]

mnist_dataset <- as.data.frame(mnist$train$images)
mnist_dataset$label <- mnist$train$label
mnist_dataset$label <- as.factor(mnist_dataset$label)
mnist_dataset # dataset final

idx_train <- sample(1250, 1000)
dtrain <- mnist_dataset[idx_train,]
dtest <- mnist_dataset[-idx_train,]

if (sum(is.na(mnist_dataset)) == 0) {
  print("No hay valores nulos")
}

########################
##  FUNCIONES UTILES  ##
########################

calcular_accuracy_rpart <- function(modelo, dtest) {
  pred <- predict(modelo, newdata=dtest, type="class")
  m_conf <- table(pred, dtest$label)
  accuracy <- sum(diag(m_conf))/sum(m_conf)
  return (accuracy)
}

calcular_accuracy_caret <- function(modelo, dtest) {
  pred <- predict(modelo, newdata=dtest)
  m_conf <- table(pred, dtest$label)
  accuracy <- sum(diag(m_conf))/sum(m_conf)
  return (accuracy)
}


############################
##  ENTRENAMIENTO MODELO  ##
############################

##### MODELO BASICO #####
modelo <- rpart(label ~ ., data=dtrain, method="class")
ac1 <- calcular_accuracy_rpart(modelo, dtest)
pred <- predict(modelo, newdata=dtest, type="class")
ac1


##### MODELO CON GRID SEARCH #####

# Para validacion cruzada
control <- trainControl(method="cv", number=8)
# Grid para parametros a probar
grid <- expand.grid(cp = seq(0.0001, 0.002, by=0.0001))

mejor_modelo <- NULL
mejor_acc <- 0
iter <- 0

for (ms in 1:5) {
  for (md in c(20, 30)) {
    control_rpart <- rpart.control(minsplit = ms, maxdepth = md)
    modelo <- train(label ~., data=dtrain, method="rpart", trControl=control, tuneGrid=grid, control=control_rpart)
    acc <- calcular_accuracy_caret(modelo, dtest)
    print(paste("Iter:", iter))
    iter <- iter + 1
    if (acc > mejor_acc) {
      mejor_acc <- acc
      mejor_modelo <- modelo
    }
  }
}
mejor_acc # 0.728 cp=8e-04, ms=2, md=30
mejor_modelo$finalModel$control
save(mejor_modelo, file = "mejor_modelo_tuneGrid.RData")


##### MODELO BOOSTING #####

modelo_boosting <- boosting(label ~., data=dtrain, boos=TRUE, mfinal=50,
                            control = rpart.control(
                            cp = 0.000,     
                            maxdepth = 30,
                            minbucket = 1
))

pred_boost <- predict(modelo_boosting, newdata=dtest)
ac_boost <- mean(pred_boost$class == dtest$label)
ac_boost
save(modelo_boosting, file = "modelo_boosting.RData")
