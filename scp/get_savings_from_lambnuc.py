import os, time
from colored import fg
B = fg(15)
C = fg(45)

def get_dir_ip():
    return open("dir_ip", "r").read()[:-1]



dir_ip = get_dir_ip()
print("dir_ip: \'"+ C + dir_ip + B +"\'")

# ejecutamos el comando remoto para obtener la lista de archivos (rutas absolutas)
print("Obteniendo rutas de ficheros remotos...")
ret = os.system("./remote_glob.sh " + dir_ip)


if ret == 0:

    # obtenemos y procesamos la lista de archivos
    temp_file = open(".out", "r")
    files = temp_file.read()

    files = files.split("\n")
    num_files = len(files)
    print("files: " + C + str(num_files) + B + "\n")

    #pruebas
    '''
    [print(f) for f in files]
    print(files[1][files[1].rfind("/")+1:])
    exit()
    '''

    # para cada archivo, intentamos copiarlo con scp a este pc. si falla, volvemos a intentarlo
    for i,f in enumerate(files,0):
        filename = f[f.rfind("/")+1:]
        ret = os.system("./remote_scp.sh " + dir_ip + " " + f + " " + filename)
        while ret != 0:
            print("[!] Error al copiar el archivo numero " + C + str(i) + B + "\nError number: " + C + str(ret) + B)
            print("Esperando 60 segundos...")
            time.sleep(60)
            print("Leyendo direccion IP del archivo...")
            dir_ip = get_dir_ip()
            time.sleep(1)
            print("Reintentando...")
            ret = os.system("./remote_scp.sh " + dir_ip + " " + f + " " + filename)
        print("\r                                                   \r" + C + str(i) + B + "/" + str(num_files) + " copiado. (" + C + str(round((i/num_files)*100,1)) + B + "%)", end='')

    print("\n--- done ---\n")



else:
    exit("error al ejecutar el comando remoto: " + fg(1) + str(ret) + fg(15))
