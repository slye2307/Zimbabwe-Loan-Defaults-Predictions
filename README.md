# Zimbabwe Loan Default Prediction App

This is a Streamlit app for estimating loan default risk for Zimbabwe banking-sector loan applications. It uses a saved CatBoost model and the same feature engineering logic used during training.

The app is meant to support credit-risk screening. It should not replace a final human credit decision.

## Live Demo

If you want to access the demo, press this link:

https://zimbabwe-loan-defaults-predictions-fndx9su4lrdlyen2xfuahg.streamlit.app/

## What The App Includes

- Single-loan risk prediction with plain-English risk factors and review steps.
- Batch CSV scoring for files with the same columns as `Test.csv`.
- Portfolio dashboard charts from `Train.csv`.
- Optional AI assistant, Brighty, for explaining scores and review next steps.

## Project Structure

```text
zindi/
|-- app.py                  # Main Streamlit application
|-- model.pkl               # Saved CatBoost model
|-- features.pkl            # Saved feature list used by the model
|-- Train.csv               # Training data used for dashboard charts
|-- requirements.txt        # Python packages needed to run the app
|-- runtime.txt             # Python version for Streamlit Cloud
|-- .streamlit/
|   `-- config.toml         # App theme settings
|-- README.md               # Setup and usage guide
`-- .gitignore              # Files GitHub should ignore
```

## How To Run Locally

Open a terminal in this project folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

After Streamlit starts, open the local URL it prints, usually:

```text
http://localhost:8501
```

## Required Files

The app needs these files in the same folder as `app.py`:

- `model.pkl`
- `features.pkl`
- `Train.csv`

Without them, the model or dashboard will not load.

Before making the repository public, confirm that the competition rules allow you to publish `Train.csv`. If not, keep the data private and publish only the code plus instructions for users to place the dataset beside `app.py`.

## Optional AI Chat

The `AI Assistant` page works in built-in guide mode by default. For full AI responses, add an OpenAI API key as a Streamlit secret or environment variable:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o-mini"
```

Locally, Streamlit reads secrets from `.streamlit/secrets.toml`. On Streamlit Cloud, add the same values in the app secrets settings. The app keeps working without these values, but the assistant will answer only common built-in guidance questions.

## How To Publish On GitHub

Create a new empty repository on GitHub, then run these commands from this project folder:

```powershell
git init
git add app.py model.pkl features.pkl Train.csv requirements.txt runtime.txt README.md .gitignore .streamlit/config.toml
git commit -m "Add Streamlit loan default prediction app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your real GitHub account and repository name.

If you use `git add .` instead, check `git status` before committing. The `.gitignore` is set up to avoid uploading virtual environments, logs, temporary training outputs, and unclean notebooks.

## Deploying To Streamlit Cloud

1. Push the project to GitHub.
2. Go to Streamlit Cloud.
3. Create a new app from your GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

Streamlit Cloud will install packages from `requirements.txt` and use the Python version from `runtime.txt`.

## Plain-English Risk Meaning

- **Low Risk** means the applicant looks less likely to default based on the entered details.
- **Medium Risk** means the application needs closer review.
- **High Risk** means the model sees stronger warning signs and a credit officer should review carefully.

These are AI screening signals, not automatic approval or rejection decisions.
