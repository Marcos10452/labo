
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("lightgbm")



kdataset       <- "./exp/FE7130/dataset_7130.csv.gz"


ksemilla_azar  <- c(111111,222222,333333,444444,555555,
                    666666,777777,888888,999999,123123)  

kcorte <- 9000


ktraining      <- c( 202101 )   #periodos en donde entreno
kfuture        <- c( 202103 )   #periodo donde aplico el modelo final


kexperimento   <- "ten_five_folds_salidas_kaggle"

kmax_bin           <-    31
klearning_rate     <-     0.010325911
knum_iterations    <-   622
knum_leaves        <-   1023
kmin_data_in_leaf  <-  294
kfeature_fraction  <-     0.524816212

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa
setwd( "~/buckets/b1" )

#cargo el dataset donde voy a entrenar
dataset  <- fread(kdataset, stringsAsFactors= TRUE)



#--------------------------------------

#paso la clase a binaria que tome valores {0,1}  enteros
#set trabaja con la clase  POS = { BAJA+1, BAJA+2 } 
#esta estrategia es MUY importante
dataset[ , clase01 := ifelse( clase_ternaria %in%  c("BAJA+2","BAJA+1"), 1L, 0L) ]

#--------------------------------------

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )

#--------------------------------------


#establezco donde entreno
dataset[ , train  := 0L ]
dataset[ foto_mes %in% ktraining, train  := 1L ]

#--------------------------------------
#creo las carpetas donde van los resultados
#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0("./exp/", kexperimento, "/" ), showWarnings = FALSE )
setwd( paste0("./exp/", kexperimento, "/" ) )   #Establezco el Working Directory DEL EXPERIMENTO



#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ train==1L, campos_buenos, with=FALSE]),
                        label= dataset[ train==1L, clase01] )

for (semilla in ksemilla_azar) {

  
modelo  <- lgb.train( data= dtrain,
                      param= list( objective=          "binary",
                                   max_bin=            kmax_bin,
                                   learning_rate=      klearning_rate,
                                   num_iterations=     knum_iterations,
                                   num_leaves=         knum_leaves,
                                   min_data_in_leaf=   kmin_data_in_leaf,
                                   feature_fraction=   kfeature_fraction,
                                   seed=               semilla
                                  )
                      )


dapply  <- dataset[ foto_mes== kfuture ]

prediccion  <- predict( modelo, 
                        data.matrix( dapply[, campos_buenos, with=FALSE ])
                        )

tb_entrega  <-  dapply[ , list( numero_de_cliente, foto_mes ) ]
tb_entrega[  , prob := prediccion ]

setorder( tb_entrega, -prob )

tb_entrega[  , Predicted := 0L ]
tb_entrega[ 1:kcorte, Predicted := 1L ]
  
fwrite( tb_entrega[ , list(numero_de_cliente, Predicted)], 
          file= paste0(  kexperimento, "_", semilla, ".csv" ),
          sep= "," )

}