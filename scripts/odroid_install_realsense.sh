#install the ubuntu 16.04 with kernel 4.14 image 

#sudo date --set "24 April 2018 11:26:35" 
sudo apt-get update
sudo apt-get upgrade
#coffee


#sudo apt-get install
#sudo apt-get upgrade
#sudo systemctl disable avahi-daemon
#sudo apt-get upgrade
#sudo systemctl enable avahi-daemon

#manually re-update boot.ini, because it will be changed
sudo nano /media/boot/boot.ini

sudo reboot

#manually upgrade packages that were left out for various reasons
sudo apt-get install libdrm-amdgpu1 libdrm-dev libdrm-etnaviv1 libdrm-exynos1 libdrm-freedreno1 libdrm-nouveau2 libdrm-omap1 libdrm-radeon1 libdrm-tegra0 libdrm2 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dri libmm-glib0 libqmi-proxy libwayland-egl1-mesa linux-headers-xu3 linux-image-xu3 mesa-vdpau-drivers modemmanager -y

sudo apt-get install git cmake git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev openssh-server libglfw3-dev gedit -y


cd ~
mkdir code -p
cd code/
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense/
mkdir build -p
cd build/
cmake ..
make -j3

sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger
sudo modprobe uvcvideo


