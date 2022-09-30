

rm( list=ls() )  #Borro todos los objetos
gc()   #Garbage Collection

require("data.table")
require("rpart")
require("ggplot2")
require("gridExtra")

euclidean_dist <- function(x, y) sqrt(sum(x)^2 + sum(y-1)^2)

roc_dist <- function(datos_roc){
  
  datos_roc[, roc_dist := mapply(euclidean_dist,datos_roc$cont,datos_roc$baja)]
  return (datos_roc)
  
}

curva_ganancia <- function(datos_roc, ganancia=78000,perdida=-2000){
  
  datos_roc[, ganancia :=  cumsum(ifelse(clase01 == 1, ganancia,perdida))]
  return (datos_roc)
  
}

best_cutoff <- function(dataset, prediccion, forecasted_fotomes = "202103"){
  
  data <- dataset[foto_mes == forecasted_fotomes]
  
  datos_roc <- data[,list(numero_de_cliente, foto_mes, clase01)]
  datos_roc [ , prob := prediccion[,"1"]]
  
  setorder(datos_roc, -prob)
  
  datos_roc[, baja := cumsum( as.integer(datos_roc$clase01)) /  sum( as.integer(datos_roc$clase01)) ]
  datos_roc[, cont := cumsum( (as.integer(datos_roc$clase01) - 1 ) / ( sum( as.integer(datos_roc$clase01)  - 1) ))]
  
  return (datos_roc)
}

particionar  <- function( data,  division, agrupa="",  campo="fold", start=1, seed=NA )
{
  if( !is.na(seed) )   set.seed( seed )
  
  bloque  <- unlist( mapply(  function(x,y) { rep( y, x )} ,   division,  seq( from=start, length.out=length(division) )  ) )  
  
  data[ , (campo) :=  sample( rep( bloque, ceiling(.N/length(bloque))) )[1:.N],
        by= agrupa ]
}


#########################################################################
## ACA ARRANCA

kganancia <- 78000
kperdida <- -2000

setwd("/home/marcos/DataScience/Curso/MdD/") ## <----------- cambiar el wd

dataset  <- fread("./datasets/competencia1_2022.csv" )

dataset  <- dataset[ foto_mes==202101 ]  #defino donde voy a entrenar


particionar( dataset, division=c(1,1), agrupa="clase_ternaria", seed= 12345 ) 
dataset[ , clase01 :=  ifelse( clase_ternaria=="BAJA+2", "1", "0" ) ]


set.seed(12345)
modelo  <- rpart(formula=   "clase01 ~ . -fold - clase_ternaria",
                 data=      dataset[ fold==1, ], 
                 xval=          0,
                 cp=           -1,
                 minsplit=    653,
                 minbucket=    311,
                 maxdepth=     7  )



prediccion  <- predict( modelo,   
                        dataset,  
                        type= "prob") 



datos_roc <- best_cutoff(dataset,prediccion,"202101")
datos_roc <- roc_dist(datos_roc)
ganancia_acumulada <- curva_ganancia(datos_roc,kganancia,kperdida)

idx_roc_min <- which.min(datos_roc$roc_dist)
idx_max_gan <- which.max(datos_roc$ganancia)

### FIGURA CURVA ROC Y DISTANCIA AL (0,1)
fig1 <- ggplot(datos_roc,aes(cont,
                     baja)
               )          + 
        geom_line()       +
        geom_point(aes( x = cont[idx_roc_min], 
                        y = baja[idx_roc_min]
                       ), 
                   data   = datos_roc, 
                   size   = 10, 
                   shape  = 1,
                   stroke = 2,
                   color  = "red"
                   )       +
        xlab("% Continua") +
        ylab("% Baja")     +
        ggtitle("Curva ROC")
  

fig2 <- ggplot(datos_roc,aes(cont,
                     roc_dist)
               )          + 
        geom_line()       +
        geom_point(aes( x = cont[idx_roc_min], 
                        y = roc_dist[idx_roc_min]
                       ), 
                   data   = datos_roc, 
                   size   = 10, 
                   shape  = 1,
                   stroke = 2,
                   color  = "red"
                   )        +

        xlab("% Continua")  +
        ylab("Distancia")   +
        ggtitle("Distancia al punto (0,1)")

grid.arrange(fig1, fig2, nrow = 2)


## CURVA DE GANANCIA
ggplot(ganancia_acumulada,aes(1:length(ganancia),
             ganancia)
       )      + 
  geom_line() +
  geom_point( 
             data=ganancia_acumulada, 
             aes(x   = idx_roc_min, 
                 y   = ganancia[idx_roc_min],
                 col = "red"),
             size    = 7, 
             shape   = 1,
             stroke  = 2,
             show.legend = T) +

  geom_point( 
             data    = ganancia_acumulada, 
             aes(x   = idx_max_gan, 
                 y   = ganancia[idx_max_gan],
                 col = "blue",),
             size    = 7, 
             shape   = 1,
             stroke  = 2,
             
             show.legend = T) +
  scale_color_manual(values = c("blue","red"), label = c("Max Ganancia","Min Dist (0,1)")) +
  xlab("Total predicciones")  +
  ylab("Ganancia Acumulada")   +
  ggtitle("Ganancia Acumulada")



### FIGURA CURVA ROC Y DISTANCIA AL (0,1)
fig3 <- ggplot(datos_roc,aes(cont,
                             baja)
                )          + 
        geom_line()       +
        geom_point(aes( x = cont[idx_roc_min], 
                        y = baja[idx_roc_min],
                        col = "red"), 
                  data   = datos_roc, 
                  size   = 7, 
                  shape  = 1,
                  stroke = 2,
                  show.legend = T
                  )       +
        geom_point( 
                  data=ganancia_acumulada, 
                  aes(x   = cont[idx_max_gan], 
                      y   = baja[idx_max_gan],
                      col = "blue"),
                  size    = 7, 
                  shape   = 1,
                  stroke  = 2,
                  show.legend = T) + 
        xlab("% Continua") +
        ylab("% Baja")     +
        ggtitle("Curva ROC") +
        scale_color_manual(values = c("blue","red"), 
                     label = c("Max Ganancia","Min Dist (0,1)")) 


fig4 <- ggplot(datos_roc,aes(cont,
                             roc_dist)
               )          + 
        geom_line()       +
        geom_point(aes( x = cont[idx_roc_min], 
                        y = roc_dist[idx_roc_min],
                        col = "red"
                        ), 
                  data   = datos_roc, 
                  size   = 7, 
                  shape  = 1,
                  stroke = 2,
                  show.legend = T
                  )      +
        geom_point( 
                    data=ganancia_acumulada, 
                    aes(x   = cont[idx_max_gan], 
                        y   = roc_dist[idx_max_gan],
                        col = "blue"),
                    size    = 7, 
                    shape   = 1,
                    stroke  = 2,
                    show.legend = T) +          
        xlab("% Continua")  +
        ylab("Distancia")   +
        ggtitle("Distancia al punto (0,1)") +
        scale_color_manual(values = c("blue","red"), 
                            label = c("Max Ganancia","Min Dist (0,1)")) 

grid.arrange(fig3, fig4, nrow = 2)


paste0("MAX GANANCIA en la prediccion " , idx_max_gan)


