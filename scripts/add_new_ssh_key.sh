ssh-keygen -t ed25519 -C "je@moeder.com"
eval "$(ssh-agent -s)" #check if process exists
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
#copy paste that, go to github, settings, add new ssh key
