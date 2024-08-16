apt-get update
apt install git
pip install --upgrade pip
pip install -r requirements.txt

git pull origin main
python main.py