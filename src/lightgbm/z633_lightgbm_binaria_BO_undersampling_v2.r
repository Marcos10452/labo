#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")

require("lightgbm")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++ VARIABLES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#kdirectoriotrabajo<-"~/buckets/b1/" #Directorio de trabajo
#kdirectortiodataset<-"./exp/FE7130/dataset_7130.csv.gz"   #Directorio de dataset y archivo datase
#kdirectortioexp<-"./exp/"  #Directorio donde queda el experimiento
#kexperimento   <- "M1-testganancia_100%"                       #Nombre del experimiento

kdirectoriotrabajo<-"/home/marcos/DataScience/Curso/MdD/" #Directorio de trabajo
kdirectortiodataset<-"./datasets/competencia1_2022.csv"   #Directorio de dataset y archivo datase
kdirectortioexp<-"./labo/Experimento_Undersampling/exp/"  #Directorio donde queda el experimiento
kexperimento   <- "testganancia_80"                       #Nombre del experimiento

# ATENCION  si NO se quiere utilizar  undersampling  se debe  usar  kundersampling <- 1.0
kundersampling  <-1.0   # un undersampling de 0.1  toma solo el 10% de los CONTINUA


#Vector semilla
vsemilla_azar  <- c(757577, 333563, 135719, 101009, 531143)  #Aqui poner la propia semilla
#vsemilla_azar  <- c(7575773)  #Aqui poner la propia semilla

#Mes que se corre
ktraining      <- c( 202101 )   #periodos en donde entreno
kfuture        <- c( 202103 )   #periodo donde aplico el modelo final


#1531	TRUE	999983	0.010253295899519	0.229763939138759	7093	285	0.020169638283811	25740000	84
#................... Hiperparametros....................
knum_iterations    <-   873
kmax_bin           <-    31
klearning_rate     <-     0.0207719898579169
kfeature_fraction  <-  0.84789114356653
kmin_data_in_leaf  <-  6714
knum_leaves        <-   267
kprob_corte        <-  0.0327131986372104
#......................................................
#<<<<<<<<<<<<<<<<<<<<<<<Normalización hp>>>>>>>>>>>>>>>

kmin_data_in_leaf<-as.integer(kmin_data_in_leaf*kundersampling)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>

kPOS_ganancia  <- 78000
kNEG_ganancia  <- -2000

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#------------------------------------- Funciones -------------------------------------------------------------
#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos

loguear  <- function( reg, arch=NA, folder=kdirectortioexp, ext=".txt", verbose=TRUE )
{
  archivo  <- arch
  if( is.na(arch) )  archivo  <- paste0(  folder, substitute( reg), ext )

  if( !file.exists( archivo ) )  #Escribo los titulos
  {
    linea  <- paste0( "fecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )

    cat( linea, file=archivo )
  }

  linea  <- paste0( format(Sys.time(), "%Y%m%d %H%M%S"),  "\t",     #la fecha y hora
                    gsub( ", ", "\t", toString( reg ) ),  "\n" )

  cat( linea, file=archivo, append=TRUE )  #grabo al archivo

  if( verbose )  cat( linea )   #imprimo por pantalla
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#esta funcion calcula internamente la ganancia de la prediccion probs

fganancia_logistic_lightgbm   <- function( probs, datos) 
{
  vlabels  <- get_field(datos, "label")
  vpesos   <- get_field(datos, "weight")
  
  gan  <- sum( (probs > PROB_CORTE  ) *
                 ifelse( vpesos == 1.0000002, kPOS_ganancia, 
                         ifelse( vpesos == 1.0000001, kNEG_ganancia, kNEG_ganancia / kundersampling ) ) )
  
  
  return( list( "name"= "ganancia", 
                "value"=  gan,
                "higher_better"= TRUE ) )
}
#------------------------------------------------------------------------------
#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales, la semilla del mal ...

EstimarGanancia_lightgbm  <- function( x, ksemilla_azar )
{
  gc()  #libero memoria

  #llevo el registro de la iteracion por la que voy
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1

  PROB_CORTE <<- x$prob_corte   #asigno la variable global

  kfolds  <- 5   # cantidad de folds para cross validation

  param_basicos  <- list( objective= "binary",
                          metric= "custom",
                          first_metric_only= TRUE,
                          boost_from_average= TRUE,
                          feature_pre_filter= FALSE,
                          verbosity= -100,
                          seed= ksemilla_azar,
                          max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
                          min_gain_to_split= 0.0, #por ahora, lo dejo fijo
                          lambda_l1= 0.0,         #por ahora, lo dejo fijo
                          lambda_l2= 0.0,         #por ahora, lo dejo fijo
                          #num_iterations= 9999,    #un numero muy grande, lo limita early_stopping_rounds
                          force_row_wise= TRUE    #para que los alumnos no se atemoricen con tantos warning
                        )

  #el parametro discolo, que depende de otro
  param_variable  <- list(  early_stopping_rounds= as.integer(50 + 5/x$learning_rate) )

  param_completo  <- c( param_basicos, param_variable, x )

  set.seed( ksemilla_azar )
  modelocv  <- lgb.cv( data= dtrain,
                       eval= fganancia_logistic_lightgbm,
                       stratified= TRUE, #sobre el cross validation (estratificar los datos)
                       nfold= kfolds,    #folds del cross validation
                       param= param_completo,
                       verbose= 100
                      )

  #obtengo la ganancia
  ganancia  <- unlist(modelocv$record_evals$valid$ganancia$eval)[ modelocv$best_iter ]

  ganancia_normalizada  <-  ganancia* kfolds     #normailizo la ganancia

  #el lenguaje R permite asignarle ATRIBUTOS a cualquier variable
  attr(ganancia_normalizada ,"extras" )  <- list("num_iterations"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra

  param_completo$num_iterations <- modelocv$best_iter  #asigno el mejor num_iterations
  param_completo["early_stopping_rounds"]  <- NULL     #elimino de la lista el componente  "early_stopping_rounds"

  #logueo 
  xx  <- param_completo
  xx$ganancia  <- ganancia_normalizada   #le agrego la ganancia
  xx$iteracion <- GLOBAL_iteracion
  loguear( xx, arch= klog )

  return( ganancia_normalizada )
}
##--------------------------------------------------------------------------------------------------------------
#Aqui empieza el programa
#Aqui se debe poner la carpeta de la computadora local
setwd(kdirectoriotrabajo)

#cargo el dataset donde voy a entrenar el modelo
dataset  <- fread(kdirectortiodataset)
#dataset<-dataset[1:3000,]

#--------------------------------------
#creo las carpetas donde van los resultados
#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( kdirectortioexp,  showWarnings = FALSE ) 
dir.create( paste0(kdirectortioexp, kexperimento, "/" ), showWarnings = FALSE )
setwd( paste0(kdirectortioexp, kexperimento, "/" ) )   #Establezco el Working Directory DEL EXPERIMENTO

#en estos archivos quedan los resultados
klog        <- kexperimento



GLOBAL_iteracion  <- 0   #inicializo la variable global

#si ya existe el archivo log, traigo hasta donde llegue
if( file.exists(klog) )
{
  tabla_log  <- fread( klog )
  GLOBAL_iteracion  <- nrow( tabla_log )
}


#Aquí se crea columna clase01 donde todos los BAJA+2 y BAJA+1 se pasan a '1' y los CONTINUA a '0'
#paso la clase a binaria que tome valores {0,1}  enteros. clase01 es binaria por eso entre '0' y '1'
dataset[ foto_mes %in% ktraining, clase01 := ifelse( clase_ternaria=="CONTINUA", 0L, 1L) ]


#En "campos_buenos" se ingresa el nombre de todas las variables (columnas) del dataset y 
# luego se agregan 4 columnas adicionales "clase_ternaria","clase01", "azar"y  "training"
#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01", "azar", "training" ) )

set.seed( vsemilla_azar[i] )

#Se genera la columna "azar" en el dataset
#runif(n) generates n uniform random numbers between 0 and 1.
# R: generate uniform randum numbers with runif(…)
dataset[  , azar := runif( nrow( dataset ) ) ]
# Se genera la columna training con todos valores '0'
dataset[  , training := 0L ]
#Acá se debe cumplir que foto_mes sea ktraining 'AND'  
# (si azar <= al kundersampling 'OR' campo clase_ternaria es "BAJA+1" o "BAJA+2" ) se hace campo training = 1 (entero).
dataset[ foto_mes %in% ktraining & ( azar <= kundersampling | clase_ternaria %in% c( "BAJA+1", "BAJA+2" ) ), training := 1L ]

#dejo los datos en el formato que necesita LightGBM
#data	
#a matrix object, a dgCMatrix object, a character representing a path to a text file (CSV, TSV, or LibSVM), or a character representing a path to a binary lgb.Dataset file

#label	
#vector of labels to use as the target variable

#weight	
#numeric vector of sample weights: Aquí le dió mas "peso" a BAJA+2, luego a BAJA+1 y al final a CONINUA

#free_raw_data	
#LightGBM constructs its data format, called a "Dataset", from tabular data. By default, that Dataset object on the R side does not keep a copy of the raw 
#data. This reduces LightGBM's memory consumption, but it means that the Dataset object cannot be changed after it has been constructed.
#If you'd prefer to be able to change the Dataset object after construction, set free_raw_data = FALSE.


dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ training == 1L, campos_buenos, with=FALSE]),
                        label= dataset[ training == 1L, clase01 ],
                        weight=  dataset[ training == 1L, ifelse( clase_ternaria=="BAJA+2", 1.0000002, ifelse( clase_ternaria=="BAJA+1",  1.0000001, 1.0) )],
                        free_raw_data= FALSE  )




#Aqui se llama con los hiperparametros. Habría que 
x  <- list( "num_iterations"   = knum_iterations,
            "max_bin"          = kmax_bin,
            "learning_rate"    = klearning_rate,
            "feature_fraction" = kfeature_fraction,
            "min_data_in_leaf" = kmin_data_in_leaf,
            "num_leaves"       = knum_leaves,
            "prob_corte"       = kprob_corte )

gan_min=99999999
gan_max=0
ganacia_total=0

for (i in seq_along(vsemilla_azar))
{
    cat( vsemilla_azar[i], " semilla elegida\n")
    
    #Obtengo la ganancia estimada del 5-kfold
    ganacia_estimada<-EstimarGanancia_lightgbm( x ,vsemilla_azar[i])
  
    #Busco el minimo valor de la ganacia generada por las semilla
    if(gan_min>ganacia_estimada) {gan_min<-ganacia_estimada}
    #Busco el maximo  valor de la ganacia generada por las semilla
    if(gan_max<ganacia_estimada) {gan_max<-ganacia_estimada}

    #Acumulo las ganancias    
    ganacia_total<<-ganacia_total+ganacia_estimada

    #promedio de ganancias según vector de semillas
    ganacia_promedio<-(ganacia_total/i)
    #Imprimo
    cat("\nGanacia minima: ",gan_min )
    cat("\nGanacia maxima: ",gan_max )
    cat("\nPromedio de semillas: ", ganacia_promedio, "\n")
    
}

#lgb.create_tree_digraph (model, tree_index=0, show_info=['split_gain', 'internal_value', 'internal_count','internal_weight', 'leaf_count', 'leaf_weight', 'data_percentage'])

