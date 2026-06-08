# Installation of dvc
pip install "dvc[all]"

# Initialize Git FIRST because DVC requires a Git repo
git init

# Initialize DVC. This creates .dvc/ and .dvcignore
dvc init

# Commit the DVC scaffolding so teammates get the same setup
git add .dvc .dvcignore
git commit -m "init dvc"

# Make local remote
mkdir -p ./tmp
dvc remote add -d localremote ./tmp


# Add data and track it
dvc add data/diamonds.csv
git add data/diamonds.csv.dvc data/.gitignore
git commit -m "some comment"


# Go back to prev version
git checkout <commit_hash>
dvc checkout

# Upload cache to linked storage
dvc push

# Runn all pipelines
dvc repro

# Take a look at stuff like metrics
dvc dag
dvc metrics show

# Run inline without changing params.yaml
dvc exp run --set-param train.n_estimators=200