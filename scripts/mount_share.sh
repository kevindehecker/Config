id -g ubuntu
id -u ubuntu

sudo mkdir -p /etc/smbcredentials
sudo nano /etc/smbcredentials/houjebek

# username=<USER>
# password=<PASS>

sudo chmod 600 /etc/smbcredentials/houjebek


sudo mount -t cifs //192.168.68.54/Film /home/houjebek/shares/Film -o credentials=/etc/smbcredentials/houjebek,uid=1000,gid=1000,rw,vers=3.0
sudo mount -t cifs //192.168.68.54/Serie /home/houjebek/shares/Serie -o credentials=/etc/smbcredentials/houjebek,uid=1000,gid=1000,rw,vers=3.0
sudo mount -t cifs //192.168.68.54/Backup /home/houjebek/shares/Backup -o credentials=/etc/smbcredentials/houjebek,uid=1000,gid=1000,rw,vers=3.0

# or put in /etc/fstab:
# //192.168.68.54/Film /home/houjebek/shares/Film cifs credentials=/etc/smbcredentials/houjebek,uid=1000,gid=1000,vers=3.0 0 0
# ...# 

# similar for pats:
sudo mkdir -p /etc/smbcredentials
sudo nano /etc/smbcredentials/kevin
sudo chmod 600 /etc/smbcredentials/kevin


sudo mount -t cifs //192.168.68.54/Film /home/houjebek/shares/Monitoring -o credentials=/etc/smbcredentials/kevin,uid=1000,gid=1000,rw,vers=3.0
sudo mount -t cifs //192.168.68.54/BagLogs /home/houjebek/shares/BagLogs -o credentials=/etc/smbcredentials/kevin,uid=1000,gid=1000,rw,vers=3.0
sudo mount -t cifs //192.168.68.54/Backup /home/houjebek/shares/Backup -o credentials=/etc/smbcredentials/kevin,uid=1000,gid=1000,rw,vers=3.0

