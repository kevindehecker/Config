id -g ubuntu
id -u ubuntu

sudo mount -t cifs //192.168.1.11/jetserketser -o username=houjebek,password=<PASSWORD>,uid=1000,gid=1000 hv

sudo mount -t cifs -o vers=1.0 //192.168.1.11/jetserketser -o username=houjebek,password=pass,uid=1000,gid=1000 hv
