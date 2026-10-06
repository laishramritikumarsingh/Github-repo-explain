# RepoLens AI — Streamlit Deployment Package

This package is designed for direct deployment on Streamlit Community Cloud.

## Entry point

Use:

```text
streamlit_app.py
```

The Streamlit frontend and repository-analysis backend are contained in this single file. There is no separate FastAPI process to start.

## Requirements

The app intentionally uses only:

```text
streamlit==1.49.1
requests==2.32.3
```

Pydantic, FastAPI, Uvicorn, PyTorch, Transformers, GitPython, and local model runtimes are not required.

## Deploy on Streamlit Community Cloud

1. Upload or push the project files to GitHub.
2. Create a new Streamlit app.
3. Select `streamlit_app.py` as the main file.
4. Use Python 3.12 or another currently supported Python version from Streamlit's Advanced settings.
5. Deploy.

The app works without an API secret by using the built-in structural analyzer. For AI-generated explanations, add this secret in Streamlit:

```toml
HF_TOKEN = "hf_your_token_here"
HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct:fastest"
```

The Hugging Face call uses the current router-based chat-completions endpoint.

## Local run

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Notes

Only public GitHub repositories are supported. Large repositories are limited to keep the cloud app responsive and avoid excessive memory usage.
