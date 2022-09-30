# LightGBM  cambiando algunos de los parametros

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("lightgbm")

#Aqui se debe poner la carpeta de la computadora local
setwd("~/buckets/b1/")   #Establezco el Working Directory

#cargo el dataset donde voy a entrenar
dataset  <- fread("./datasets/dataset_7130.csv.gz", stringsAsFactors= TRUE)


#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ foto_mes==202101, clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]


#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )


#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ foto_mes==202101, campos_buenos, with=FALSE]),
                        label= dataset[ foto_mes==202101, clase01] )

#genero el modelo con los parametros por default
modelo  <- lgb.train( data= dtrain,
                      param= list( objective=        "binary",
                                   max_bin=             31,
                                   learning_rate=        0.0317210985800962,
                                   num_iterations=      714,
                                   num_leaves=         39,
                                   feature_fraction=     0.781037448186109,
                                   min_data_in_leaf=  2761,
                                   seed=            762101 )  )


#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dataset[ foto_mes==202103, campos_buenos, with=FALSE ]) )


#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dataset[ foto_mes==202103, numero_de_cliente],
                                 "Predicted"= as.integer(prediccion > 0.0293778182433488 ) )  ) #genero la salida

dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/KA5120/", showWarnings = FALSE )
archivo_salida  <- "./exp/KA5120/KA5120_Exp_01b.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep= "," )


#ahora imprimo la importancia de variables
#tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
#archivo_importancia  <- "./exp/KA5120/5120_importancia_001.txt"

#fwrite( tb_importancia, 
#        file= archivo_importancia, 
#        sep= "\t" )

