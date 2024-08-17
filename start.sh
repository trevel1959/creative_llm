apt-get update
apt-get install git openssh-client -y
ssh-keygen -t ed25519 -C “jrjh0415@gmail.com”
eval "$(ssh-agent -s)"
cat /root/.ssh/id_ed25519.pub