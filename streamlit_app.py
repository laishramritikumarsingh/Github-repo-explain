from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st

APP_TITLE = "RepoLens AI"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct:fastest"
MAX_FILE_BYTES = 500_000
MAX_FILES_TO_ANALYZE = 18
MAX_CONTEXT_CHARS = 24_000
MAX_ARCHIVE_BYTES = 25_000_000
REQUEST_TIMEOUT = (10, 60)

IGNORED_DIRS = {
    ".git", "venv", ".venv", "env", "node_modules", "__pycache__", "dist",
    "build", "coverage", ".next", ".idea", ".pytest_cache", "target", "vendor",
}
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".html", ".css", ".scss", ".sql", ".kt", ".swift",
    ".json", ".yaml", ".yml", ".toml", ".md", ".txt",
}
PRIORITY = {
    "readme.md": 0,
    "readme": 1,
    "streamlit_app.py": 2,
    "main.py": 3,
    "app.py": 4,
    "server.py": 5,
    "requirements.txt": 6,
    "pyproject.toml": 7,
    "package.json": 8,
    "dockerfile": 9,
}

SYSTEM_PROMPT = """You are a software project explainer for a college student.
Analyze only the supplied repository context. Do not invent features that are not supported by the files.
Do not reproduce large code blocks. Use simple English and concise Markdown.
Return these sections:
# Project Overview
# Main Features
# Project Structure
# How the Application Works
# Technologies Used
# Important Code Components
# Simple Summary
"""


def get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value.strip()
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value).strip()


def parse_github_url(repo_url: str) -> tuple[str, str]:
    parsed = urlparse((repo_url or "").strip())
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Enter a public GitHub URL such as https://github.com/owner/repository")
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) != 2:
        raise ValueError("Use a repository URL in the form https://github.com/owner/repository")
    owner, repo = parts[0], re.sub(r"\.git$", "", parts[1])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("The GitHub repository URL contains invalid characters.")
    return owner, repo


def download_public_repository(repo_url: str) -> tuple[str, bytes]:
    """Download a public GitHub repository as a ZIP without requiring GitPython or a local git server."""
    owner, repo = parse_github_url(repo_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "RepoLens-AI-Streamlit"}

    api_response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
    if api_response.status_code == 404:
        raise ValueError("Repository was not found. Make sure it is public and the URL is correct.")
    api_response.raise_for_status()
    repo_info = api_response.json()
    branch = repo_info.get("default_branch") or "main"

    archive_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    archive_response = requests.get(
        archive_url,
        headers={"User-Agent": "RepoLens-AI-Streamlit"},
        timeout=REQUEST_TIMEOUT,
    )
    archive_response.raise_for_status()
    if len(archive_response.content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Repository archive is too large for this cloud app. Try a smaller repository.")
    return repo, archive_response.content


def should_include_file(path: PurePosixPath, size: int) -> bool:
    if size > MAX_FILE_BYTES:
        return False
    if any(part in IGNORED_DIRS or part.startswith(".") for part in path.parts[:-1]):
        return False
    name = path.name.lower()
    return path.suffix.lower() in SUPPORTED_EXTENSIONS or name in PRIORITY or name.endswith(".md")


def build_repository_context(repo_name: str, archive_bytes: bytes) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("GitHub returned an invalid repository archive.") from exc

    members = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        raw = info.filename.replace("\\", "/").lstrip("/")
        parts = raw.split("/", 1)
        relative = PurePosixPath(parts[1] if len(parts) == 2 else parts[0])
        if str(relative) and should_include_file(relative, info.file_size):
            members.append((info, relative))

    tree = [str(path) for _, path in members[:120]]
    selected = sorted(
        members,
        key=lambda pair: (
            PRIORITY.get(pair[1].name.lower(), 50),
            len(pair[1].parts),
            str(pair[1]).lower(),
        ),
    )[:MAX_FILES_TO_ANALYZE]

    chunks: list[str] = []
    included: list[str] = []
    total_chars = 0
    for info, relative in selected:
        try:
            raw_bytes = archive.read(info)
            text = raw_bytes.decode("utf-8", errors="replace").strip()
        except (KeyError, OSError):
            continue
        if not text:
            continue
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        text = text[: min(5_000, remaining)]
        chunks.append(f"===== FILE: {relative} =====\n{text}")
        included.append(str(relative))
        total_chars += len(text)

    if not chunks:
        raise ValueError("No readable source or documentation files were found in the repository.")

    return {
        "repository_name": repo_name,
        "file_count": len(members),
        "analyzed_file_count": len(included),
        "important_files": included,
        "structure": tree,
        "context_text": "\n\n".join(chunks),
    }


def make_prompt(context: dict[str, Any]) -> str:
    structure = "\n".join(context["structure"])
    files = ", ".join(context["important_files"])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Repository: {context['repository_name']}\n"
        f"Files discovered: {context['file_count']}\n"
        f"Files analyzed: {context['analyzed_file_count']}\n"
        f"Included files: {files}\n\n"
        f"Repository structure:\n{structure}\n\n"
        f"Repository context:\n{context['context_text']}"
    )


def extract_summary_fallback(context: dict[str, Any]) -> str:
    files = context["important_files"]
    extensions = sorted({PurePosixPath(name).suffix.lower() for name in files if PurePosixPath(name).suffix})
    languages = []
    mapping = {
        ".py": "Python", ".js": "JavaScript", ".jsx": "React/JavaScript", ".ts": "TypeScript",
        ".tsx": "React/TypeScript", ".java": "Java", ".cpp": "C++", ".c": "C",
        ".cs": "C#", ".go": "Go", ".rs": "Rust", ".php": "PHP", ".html": "HTML",
        ".css": "CSS", ".scss": "SCSS", ".sql": "SQL", ".json": "JSON", ".md": "Markdown",
        ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
    }
    for ext in extensions:
        if ext in mapping and mapping[ext] not in languages:
            languages.append(mapping[ext])
    lang_text = ", ".join(languages) or "the detected source files"
    structure = "\n".join(f"- `{name}`" for name in files[:12])
    return f"""# Project Overview
**{context['repository_name']}** is a public GitHub repository analyzed by RepoLens AI. The app inspected {context['file_count']} readable project files and selected {context['analyzed_file_count']} important files for analysis.

# Main Features
- Public GitHub repository analysis
- Automatic selection of important source and documentation files
- Simple project structure and technology summary
- Optional AI-generated explanation

# Project Structure
{structure}

# How the Application Works
1. The user enters a public GitHub repository URL.
2. The app downloads the repository archive.
3. It filters out generated folders and very large files.
4. It selects key files and prepares a compact context.
5. It generates an explanation with an AI provider when available; otherwise it provides this automatic structural summary.

# Technologies Used
{lang_text}
- Streamlit
- Python standard library
- Requests

# Important Code Components
- `streamlit_app.py` contains both the Streamlit interface and the repository-analysis backend functions.
- GitHub repository parsing and archive download are handled directly in Python.
- The analysis layer builds a limited context so the app does not send an entire large repository to the AI model.

# Simple Summary
RepoLens AI turns a public GitHub repository into an easy-to-understand project explanation without requiring a separate FastAPI server.
"""


def generate_ai_explanation(context: dict[str, Any]) -> tuple[str, str, bool]:
    token = get_secret("HF_TOKEN")
    if not token:
        return extract_summary_fallback(context), "Built-in project analyzer", False

    model = get_secret("HF_MODEL", DEFAULT_MODEL)
    prompt = make_prompt(context)
    endpoint = "https://router.huggingface.co/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 900,
        "stream": False,
    }
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        # Keep the app usable even if a model/provider is temporarily unavailable.
        return extract_summary_fallback(context), "Built-in project analyzer", False

    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return extract_summary_fallback(context), "Built-in project analyzer", False
    message = choices[0].get("message", {})
    text = str(message.get("content", "")).strip()
    if not text:
        return extract_summary_fallback(context), "Built-in project analyzer", False
    return text, model, True


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")

    st.title("🔎 RepoLens AI")
    st.caption("Understand a public GitHub codebase in simple language")

    with st.sidebar:
        st.subheader("How it works")
        st.write("1. Enter a public GitHub repository.")
        st.write("2. RepoLens downloads and scans the source files.")
        st.write("3. It builds a compact code context.")
        st.write("4. AI explains the project when HF_TOKEN is available.")
        st.write("5. Without a token, a built-in summary still works.")
        st.divider()
        st.caption("Cloud-safe design: no FastAPI server, no local model download, and no Pydantic dependency.")

    url = st.text_input(
        "GitHub repository URL",
        placeholder="https://github.com/psf/requests",
        help="The repository must be public.",
    )

    analyze = st.button("Analyze Repository", type="primary", use_container_width=True)
    if not analyze:
        st.info("Start with a small public GitHub repository.")
        return

    if not url.strip():
        st.warning("Please enter a GitHub repository URL.")
        return

    with st.spinner("Downloading and analyzing repository..."):
        try:
            repo_name, archive = download_public_repository(url)
            context = build_repository_context(repo_name, archive)
        except (requests.RequestException, ValueError) as exc:
            st.error(str(exc))
            return

    with st.spinner("Generating project explanation..."):
        explanation, model_name, used_ai = generate_ai_explanation(context)

    st.success("Analysis completed")
    a, b, c = st.columns(3)
    a.metric("Files discovered", context["file_count"])
    b.metric("Files analyzed", context["analyzed_file_count"])
    c.metric("Explanation", "AI" if used_ai else "Built-in")
    st.caption(f"Model/source: `{model_name}`")

    st.markdown(explanation)
    st.download_button(
        "Download explanation",
        data=explanation,
        file_name=f"{context['repository_name']}_explanation.md",
        mime="text/markdown",
        use_container_width=False,
    )

    with st.expander("Files analyzed"):
        st.code("\n".join(context["important_files"]))


if __name__ == "__main__":
    main()
