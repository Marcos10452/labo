# source( "~/labo/src/lightgbm/z633_lightgbm_binaria_BO.r" )
# Este script esta pensado para correr en Google Cloud
#   8 vCPU
#  64 GB memoria RAM
# 256 GB espacio en disco

# se entrena con POS =  { BAJA+1, BAJA+2 }
# Optimizacion Bayesiana de hiperparametros de  lightgbm, con el metodo TRADICIONAL de los hiperparametros originales de lightgbm
# 5-fold cross validation
# la probabilidad de corte es un hiperparametro

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")

require("lightgbm")


#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++ VARIABLES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#kdirectoriotrabajo<-"~/buckets/b1/" #Directorio de trabajo
#kdataset<-"./exp/FE7130/dataset_7130.csv.gz"   #Directorio de dataset y archivo datase
#kdirectortioexp<-"./exp/"  #Directorio donde queda el experimiento
#kexperimento   <- "M1-testganancia_100%"                       #Nombre del experimiento

kdirectoriotrabajo<-"/home/marcos/DataScience/Curso/MdD/" #Directorio de trabajo
kdirectortiodataset<-"./datasets/competencia1_2022.csv"   #Directorio de dataset y archivo datase
kdirectortioexp<-"./exp/"  #Directorio donde queda el experimiento
kexperimento   <- "testganancia_80"                       #Nombre del experimiento


#Vector semilla
#vsemilla_azar  <- c(757577, 333563, 135719, 101009, 531143)  #Aqui poner la propia semilla
ksemilla_azar  <- c(7575773)  #Aqui poner la propia semilla

#Mes que se corre
ktraining      <- c( 202101 )   #periodos en donde entreno
kfuture        <- c( 202103 )   #periodo donde aplico el modelo final


# ATENCION  si NO se quiere utilizar  undersampling  se debe  usar  kundersampling <- 1.0
kundersampling  <- 1   # un undersampling de 0.1  toma solo el 10% de los CONTINUA

prob_min  <- 0.5/( 1 + kundersampling*39)
prob_max  <- pmin( 1.0, 4/( 1 + kundersampling*39) )

kPOS_ganancia  <- 78000
kNEG_ganancia  <- -2000

kBO_iter  <- 10   #cantidad de iteraciones de la Optimizacion Bayesiana


#<<<<<<<<<<<<<<<<<<<<<<<<<< Lectura dataset >>>>>>>>>>>>>>>>>>>>>>>>
#Aqui se debe poner la carpeta de la computadora local
setwd(kdirectoriotrabajo)
#cargo el dataset donde voy a entrenar el modelo
dataset  <- fread( kdirectortiodataset)

#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0( "./exp/", kexperimento, "/"), showWarnings = FALSE )
setwd( paste0( "./exp/", kexperimento, "/") )   #Establezco el Working Directory DEL EXPERIMENTO

#en estos archivos quedan los resultados
kbayesiana  <- paste0( kexperimento, ".RDATA" )
klog        <- paste0( kexperimento, ".txt" )
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>

#................... Hiperparametros....................
dataset<- dataset[1:6000,]
dataset_size<-nrow(dataset) # Tamño filas dataset



#uso la relación que queire del dataset para el "min_data_in_leaf"
min_data_size_low=0.008  #pasarlo a %
min_data_size_high=0.05

#Calculo para poder pasarlo al parametro de mbo
lower_min_data=as.integer(dataset_size*min_data_size_low)
upper_min_data=as.integer(dataset_size*min_data_size_high)


#Coverage factor, que uso para achicar el num_leaves en base a los min_data_in_leaf seleccionados en funcion del tamaño del dataset.
#PReviene el overfiting
coverage_porcentage     <-   1

#caclulo con el coverage
lower_num_leaves=as.integer(dataset_size/upper_min_data)
upper_num_leaves=as.integer(dataset_size/lower_min_data )

#......................................................

#Aqui se cargan los hiperparametros

hs <- makeParamSet( 
  makeNumericParam("learning_rate",    lower=  0.01   , upper=    0.3),
  makeNumericParam("feature_fraction", lower=  0.2    , upper=    1.0),
  makeIntegerParam("min_data_in_leaf", lower = lower_min_data, upper = upper_min_data),
  makeIntegerParam("num_leaves", lower = as.integer(lower_num_leaves), upper =  as.integer(upper_num_leaves),trafo=function(x) as.integer(x*coverage_porcentage)),
  forbidden = expression(num_leaves > as.integer(dataset_size/min_data_in_leaf)),
  
  makeNumericParam("prob_corte",  lower= prob_min, upper= prob_max  )  #esto sera visto en clase en gran detalle
)
print(hs)

#class(min_data_in_leaf)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#------------------------------------------------------------------------------
#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos

loguear  <- function( reg, arch=NA, folder="./exp/", ext=".txt", verbose=TRUE )
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

EstimarGanancia_lightgbm  <- function( x )
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
                          max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
                          min_gain_to_split= 0.0, #por ahora, lo dejo fijo
                          lambda_l1= 0.0,         #por ahora, lo dejo fijo
                          lambda_l2= 0.0,         #por ahora, lo dejo fijo
                          max_bin= 31,            #por ahora, lo dejo fijo
                          num_iterations= 9999,   #un numero muy grande, lo limita early_stopping_rounds
                          force_row_wise= TRUE,   #para que los alumnos no se atemoricen con tantos warning
                          seed= ksemilla_azar
                        )

  #el parametro discolo, que depende de otro
  param_variable  <- list(  early_stopping_rounds= as.integer(50 + 5/x$learning_rate) )

  param_completo  <- c( param_basicos, param_variable, x )

  set.seed( ksemilla_azar )
  modelocv  <- lgb.cv( data= dtrain,
                       eval= fganancia_logistic_lightgbm,
                       stratified= TRUE, #sobre el cross validation
                       nfold= kfolds,    #folds del cross validation
                       param= param_completo,
                       verbose= -100
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

#Se genera la columna "azar" en el dataset
#runif(n) generates n uniform random numbers between 0 and 1.
# R: generate uniform randum numbers with runif(…)
set.seed( ksemilla_azar )
dataset[  , azar := runif( nrow( dataset ) ) ]
dataset[  , training := 0L ]
dataset[ foto_mes %in% ktraining & ( azar <= kundersampling | clase_ternaria %in% c( "BAJA+1", "BAJA+2" ) ), training := 1L ]

#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ training == 1L, campos_buenos, with=FALSE]),
                        label= dataset[ training == 1L, clase01 ],
                        weight=  dataset[ training == 1L, ifelse( clase_ternaria=="BAJA+2", 1.0000002, ifelse( clase_ternaria=="BAJA+1",  1.0000001, 1.0) )],
                        free_raw_data= FALSE  )



#Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar  <- EstimarGanancia_lightgbm   #la funcion que voy a maximizar

configureMlr( show.learner.output= FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo
obj.fun  <- makeSingleObjectiveFunction(
              fn=       funcion_optimizar, #la funcion que voy a maximizar
              minimize= FALSE,   #estoy Maximizando la ganancia
              noisy=    TRUE,
              par.set=  hs,     #definido al comienzo del programa
              has.simple.signature = FALSE   #paso los parametros en una lista
             )

ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)  #se graba cada 600 segundos
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )   #cantidad de iteraciones
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )

#establezco la funcion que busca el maximo
surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))

#inicio la optimizacion bayesiana
if( !file.exists( kbayesiana ) ) {
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl)
} else {
  run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista
}


quit( save="no" )

