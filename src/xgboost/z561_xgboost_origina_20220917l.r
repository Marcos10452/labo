# XGBoost  sabor original ,  cambiando algunos de los parametros

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("xgboost")

#Aqui se debe poner la carpeta de la computadora local
#setwd("~/buckets/b1/")   #Establezco el Working Directory
setwd("/home/marcos/DataScience/Curso/MdD/")
#cargo el dataset donde voy a entrenar
dataset  <- fread("./datasets/competencia1_2022.csv", stringsAsFactors= TRUE)


#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ foto_mes==202101, clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )


#dejo los datos en el formato que necesita XGBoost
dtrain  <- xgb.DMatrix( data= data.matrix(  dataset[ foto_mes==202101 , campos_buenos, with=FALSE]),
                        label= dataset[ foto_mes==202101, clase01 ] )

#genero el modelo con los parametros por default
modelo  <- xgb.train( data= dtrain,
                      param= list(objective=       "binary:logistic",
                                  gamma=               0.0,
                                  alpha=               0.0,
                                  lambda=              0.0,
                                  subsample=           1.0,
                                  #three_method="auto",
                                  grow_policy="depthwise",
                                  max_bin= 256,
                                  max_leaves= 0,
                                  scale_pos_weight=    1.0,
                                  eta=	0.01006,
                                  colsample_bytree=	0.23048,
                                  min_child_weight=	0,
                                  max_depth=	2
  
                                   ),
                      #base_score= mean( getinfo(dtrain, "label")),
                      nrounds= 2602
                    )


#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dataset[ foto_mes==202103, campos_buenos, with=FALSE ]) )


#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dataset[ foto_mes==202103, numero_de_cliente],
                                 "Predicted"= as.integer( prediccion > 0.01923 ) )  ) #genero la salida

dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/KA5610/", showWarnings = FALSE )
archivo_salida  <- "./exp/KA5610/KA5610_003.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep= "," )