ssh-keygen -t rsa -b 4096 -C "kevindehecker@hotmail.com"
eval "$(ssh-agent -s)" #check if process exists
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
#copy paste that, go to github, settings, add new ssh key