import glob
import os, json, cv2
from util import printProgressBar
from colored import fg, attr
import numpy as np

B = fg(15)
V = fg(45)

parent_folder = os.path.abspath(
    os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

dataset_dest_folder = os.path.join(parent_folder, "dataset")
raw_dataset_path = os.path.join(parent_folder, "dataset", "fase2raw")
dest_path = os.path.join(dataset_dest_folder, "examples")

# -------------  PARAMETROS  ------------
OVERRIDE_MAXVALUE = 3000  # -1 = auto detect

num_examples = 100



# ------------------------------------------------------------------------------------------------------


num_images = 0
ALLJSON = []

# contamos las imagenes totales
for json_number, json_name in enumerate(glob.glob(os.path.join(raw_dataset_path, '*.json'))):

    f = open(json_name)
    j = json.load(f)

    i = 0
    for key in j:
        num_images = num_images + 1
        i += 1
        ALLJSON.append(j[key])
    print("Numero del Json: " + V + str(json_number) + B)
    print("Imagenes en el Json: " + V + str(i) + B, end='\n\n')



print(attr(4) + "Imagenes Totales:" + attr(0) + " " + V + str(num_images) + B, end='\n\n')


num_fail = 0
num_done = 0


# Hacemos shuffle a ALLJSON para randomizar las imagenes
ALLJSON = np.array(ALLJSON)
np.random.shuffle(ALLJSON)

# Cogemos las primeras num_examples y las guardamos modificadas en la carpeta destino
for filenumber, example in enumerate(ALLJSON, 0): 
	# cargamos cada imagen
	lastSlash = example["path_depth_top_image"].rfind("/")
	justName  = example["path_depth_top_image"][lastSlash+1:]

	img_path_and_name = os.path.join(raw_dataset_path, justName)
	img = cv2.imread(img_path_and_name, flags=cv2.IMREAD_ANYDEPTH)

	# comprobamos si la imagen existe y ha sido cargada correctamente (si no, saltamos a la siguiente)
	try:
		if len(img) == 0:
			print("[!] Error al cargar la imagen: " + V + str(img_path_and_name) + B)
			exit()
	except:
		num_fail += 1
		continue  

	img = img * 30

	# obtenemos el peso
	peso = example["weight"]

	# Guardamos la imagen en la nueva carpeta con el nombre del numero mas el peso
	if not os.path.exists(dest_path):
		os.makedirs(dest_path)
	cv2.imwrite(os.path.join(dest_path, str(filenumber) + "__" + str(peso) + "Kg.png"), np.uint16(img))
	num_done += 1
	if num_done >= num_examples:
		break

print("errores: ", num_fail)

'''
# para todos los .json del raw_dataset
for json_number, json_name in enumerate(glob.glob(os.path.join(raw_dataset_path, '*.json'))):

    f = open(json_name)
    j = json.load(f)


    # estructuramos las imagenes del json
    for key in j:

        if j[key]["label"] == "lamb":
            # cargamos cada imagen
            lastSlash = j[key]["path_depth_top_image"].rfind("/")
            justName = j[key]["path_depth_top_image"][lastSlash+1:]

            img_path_and_name = os.path.join(raw_dataset_path, justName)
            img = cv2.imread(img_path_and_name, flags=cv2.IMREAD_ANYDEPTH)


            # comprobamos si la imagen existe y ha sido cargada correctamente (si no, saltamos a la siguiente)
            try:
                if len(img) == 0:
                    print("[!] Error al cargar la imagen: " + V + str(img_path_and_name) + B)
                    exit()
            except:
                num_fail += 1
                continue  

            # preparamos la imagen para ser normalizada posteriormente (todo pixel que supere OVERRIDE_MAXVALUE se queda en OVERRIDE_MAXVALUE)
            if OVERRIDE_MAXVALUE != -1:
                img = np.clip(img, None, OVERRIDE_MAXVALUE)

            # le hacemos crop
            x = 38
            y = 102
            h = 230
            w = 510
            img = img[y:y + h, x:x + w]

            # invertimos la imagen
            # if OVERRIDE_MAXVALUE != -1:
            #   img = OVERRIDE_MAXVALUE-img

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            cv2.imwrite(os.path.join(dest_path, str(num_lamb) + ".png"), np.uint16(img))

            # guardamos la etiqueta
            label = j[key]["weight"]
            label_list.append(label)
            
            num_lamb += 1

        i2 += 1
        printProgressBar(i2, num_images, prefix='Estructurando JSON ' + str(json_number) + ':', suffix='Completado',length=100,color=45)




# guardamos las etiquetas
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
d = open(os.path.join(dest_path, "labels.npy"), "wb+")
np.save(d, np.array(label_list), allow_pickle=True)
d.close()

print("\n\nEl dataset resultado se compone de " + V + str(num_lamb) + B + " imagenes.\n\n")
print(str(num_fail) + " imagenes erradas o no encontradas.")
'''
