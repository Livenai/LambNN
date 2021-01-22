from colored import fg
from cnn.nn import Model_constructor
import os


VERDE  = fg(118)
BLANCO = fg(15)
AZUL   = fg(45)
AZUL_CLARO = fg(159)


nombre_dataset = "clasiLamb_2-1_CUS"
target_size = (480, 640, 1)



parent_folder = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

################################################       PARAMETROS       ################################################


ID_MODELO = 15

# Path al modelo para cargar y reciclar
# "" para crear modelo nuevo
model_path = ""

epochs = 100
batch_size = 1  # 1

loading_batch_size = 1
learning_rate = 0.00001  # 0.00001

workers = 8  # hilos para el multiprocessing
RAM_PERCENT_LIMIT = 80  # %

# callbacks
paciencia = 500

# porcentaje de uso del dataset en entrenamiento (el resto va a la validacion)
train_percent = 0.8  # 80%





##################################################        Dicc        ##################################################
parametros = {}
parametros["id_modelo"] = ID_MODELO
parametros["epochs"] = epochs
parametros["batch_size"] = batch_size
parametros["loading_batch_size"] = loading_batch_size
parametros["learning_rate"] = learning_rate
parametros["workers"] = workers
parametros["ram_percent_limit"] = RAM_PERCENT_LIMIT
parametros["paciencia"] = paciencia
parametros["train_percent"] = train_percent

colores = {}
colores["main"] = VERDE
colores["default"] = BLANCO



##################################################        MAIN        ##################################################

# creamos el modelo (o lo cargamos si se aporta un path)
MC = Model_constructor(parent_folder, parametros, colores)
if model_path == "":
    model = MC.create_model()
else:
    model = MC.load_model_from_path(model_path)


print(colores["main"])
model.summary()
print(colores["default"])
print("#################################################################")


# compilamos (si el modelo es nuevo)
if model_path == "":
    model = MC.compile_model(model)



# obtenemos los generadores
dataset_path = os.path.join(parent_folder, "dataset", nombre_dataset)
genetators = MC.get_generators(dataset_path, target_size, data_aumentation=False)


# entrenamos
history, model = MC.fit_model(model, genetators, use_generators=True, regression=False, color=118, evaluate_each_epoch=False, restore_best_weights=True)

# evaluamos la red y mostramos los resultados
MC.print_final_classif_evaluation(model, genetators[1], target_size=target_size, num_examples=100)

# mostramos los resultados
MC.show_plot(history, regression=False, just_save=True, save_name='ClasiLamb_2-1')
MC.save_model(model, "CL2-1")



# todo -> comando para que se apague automaticamente al terminar la ejecucion (poner contraseña)
"""
sys_pass = ''
sudo = "echo \"" + sys_pass + "\" | sudo -S "  # es importante que haya un espacio despues de -S
os.system(sudo + "shutdown 0")
"""
