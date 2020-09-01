if [ $# -eq 0 ]
  then
    echo "El script necesita la direccion IP como parametro"
  else
  sshpass -p 'opticalflow' ssh -q -o StrictHostKeyChecking=no lambnuc@$1 'python3 LambNN/src/glob_savings.py' > .out
fi

