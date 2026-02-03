# Hubungkan local → GitHub
cd /Drive/D/geoscience

## sekali saja untuk init
git init
git status
git add .
git commit -m "Initial commit: geoscience toolbox"
git branch -M main
git remote add origin https://github.com/yogahariman/geoscience.git
git push -u origin main

# cara pakai
git clone https://github.com/username/geoscience.git
cd geoscience
pip install -e .



