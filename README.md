# dog-breed-bot

predict some breeds

## Telegram bot that predicts a dog’s breed from a photo

- Input: an image (any size; it is resized/cropped for the model)

- Output: a breed name (in a human-readable way)

- Model: ResNet-101 backbone + classifier head (Stanford Dogs-style labels)

- Inference: CPU-friendly via ONNX Runtime

- Artifacts management: DVC (stores model files in remote storage)

## Quick start for windows

### 1. Clone the repo

```
git clone https://github.com/saylock/dog-breed-bot.git

cd dog-breed-bot
```
### 2. Install dependencies
```
poetry install
```
### 3. Pull model artifacts
```
poetry run dvc pull
```
Now you should have:

- artifacts/pt/best.ckpt

- artifacts/onnx/model.onnx

- artifacts/class\_to\_idx.json

## Local inference:
replace "test.jpg" with the real name of your dog picture:
```
poetry run dog-breed-bot infer -i .\\test.jpg
```
or if your dog image is somewere else, use an absolut path:
```
poetry run dog-breed-bot infer -i "absolute path here"
```
Example of an output:
```
Pembroke
```
## Run the Telegram bot:

### 1. Create bot and save the token
In telegram create the bot using @BotFather, follow instructions and copy token. Create .env file in the repo root and copy token there:
```
TG_BOT_TOKEN=COPY_YOUR_TOKEN_HERE
```
Do not commit this .env file anywhere and add it to .gitignore to be sure
### 2. Start the bot and try it
```
poetry run dog-breed-bot bot
```
Then open telegram and the bot chat, send /start, then send your dog picture. the bot should reply with the breed. After using stop the bot using Control + C in the Powershell

## Optional step -- Re-Training the model
### 1. Dataset
Dataset is not included because of its largeness -- you'll have to download it from Kaggle:

https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset?resource=download and keep it locally. Unzip it under:
```
data/raw/stanford_dogs
```
### 2. Prepare data
```
poetry run dog-breed-bot prepare-data
```
Now you should have:

- data/processed/train.csv

- data/processed/val.csv

- data/processed/test.csv

- artifacts/class\_to\_idx.json

### 3. Train

**This step might take a lot of time -- I've left it over night, but you can lower the amount of epochs if the time is essential**
```
poetry run python -m dog_breed_bot.train.train epochs=30
```

Checkpoint output: artifacts/pt/best.ckpt

### 4. Export to ONNX (CPU deployment)
```
poetry run dog-breed-bot export-onnx
```
ONNX output:
artifacts/onnx/model.onnx
## Optional step -- MLflow
You can run MLflow locally to inspect experiments:
```
poetry run mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
Open in browser:

http://127.0.0.1:8080

## Repo structure:

**dog\_breed\_bot/commands.py** — CLI commands (dog-breed-bot ...)

**dog\_breed\_bot/data/prepare\_splits.py** — creates dataset manifests + balanced splits

**dog\_breed\_bot/train/** — training code

**dog\_breed\_bot/export/onnx\_export.py** — export checkpoint → ONNX

**dog\_breed\_bot/infer/onnx\_infer.py** — ONNX inference

**dog\_breed\_bot/bot/telegram\_bot.py** — Telegram bot

**artifacts/** — model artifacts (tracked by DVC)

**data/** — local datasets (ignored by Git)

## Troubleshooting:

- **dvc pull opens browser / access issues:**

complete the browser login while the command is running (if you still have troubles -- write me and I'll add you to testers)

- **bot doesn't reply:**

make sure **.env** exists and in repo root, also it should contain telegram bot token in it, restart bot
