if [ $# -lt 3 ]
  then
    echo "El script necesita la direccion IP, la ruta absoluta del archivo como parametro y el nombre del archivo a guardar"
  else
  sshpass -p 'opticalflow' scp -q -o StrictHostKeyChecking=no lambnuc@$1:$2 ./savings/$3
fi

