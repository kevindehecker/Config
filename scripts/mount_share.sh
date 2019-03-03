id -g ubuntu
id -u ubuntu

sudo mount -t cifs //192.168.1.11/jetserketser -o username=houjebek,password=<PASSWORD>,uid=1000,gid=1000 hv

sudo mount -t cifs -o vers=1.0 //192.168.1.11/jetserketser -o username=houjebek,password=pass,uid=1000,gid=1000 hv

sudo mount -t cifs //192.168.1.11/jetserketser -o username=houjebek,password=<PASSWORD>,uid=1000,gid=1000 ~/shares/jetserketser/
sudo mount -t cifs //192.168.1.13/terantisch -o username=houjebek,password=<PASSWORD>,uid=1000,gid=1000 ~/shares/terantisch/
sudo mount -t cifs //192.168.1.13/lalaladata -o username=houjebek,password=<PASSWORD>,uid=1000,gid=1000 ~/shares/lalaladata/
sudo mount -t cifs //192.168.1.13/JoeHoe -o username=houjebek,password=<PASSWORD>,uid=1000,gid=1000 ~/shares/joehoe/