# Contributing to ReviewMind

Thank you for your interest in contributing to **ReviewMind**! We welcome contributions from developers, researchers, and students in natural language processing and text data mining.

Please review this guide before submitting an issue or pull request.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to `ryadbenyakoub@gmail.com`.

---

## How Can I Contribute?

- **Reporting Bugs**: Found a bug or inconsistent behavior? Open an issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
- **Suggesting Enhancements**: Propose new feature extractors, classifiers, or evaluation techniques via our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).
- **Submitting Pull Requests**: Implement improvements, documentation updates, or unit tests.

---

## Development Workflow

### 1. Fork and Clone
Fork the repository on GitHub and clone your fork locally:

```bash
git clone https://github.com/<your-username>/ReviewMind.git
cd ReviewMind
```

### 2. Set Up Your Environment
Ensure you have Python 3.10+ installed. Install project dependencies:

```bash
pip install -r requirements.txt
```

*(Optional)* Create a `.env` file from the provided template:

```bash
# On Linux / macOS
cp .env.example .env

# On Windows PowerShell
Copy-Item .env.example .env
```

### 3. Create a Topic Branch
Create a descriptive branch for your work:

```bash
git checkout -b feature/improved-tokenization
# or
git checkout -b bugfix/fix-empty-string-handling
```

### 4. Coding Standards
- **Style**: Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- **Portability**: Use `pathlib.Path` for all filesystem paths. Never hardcode absolute system paths (e.g., `C:\...`).
- **Encoding**: Ensure all file operations explicitly declare `encoding="utf-8"` (or `latin1` where required for legacy datasets).
- **Docstrings**: Document functions and classes with descriptive docstrings detailing parameters, return types, and potential exceptions.
- **Reproducibility**: Set explicit random seeds (`random_state=42`) when training models or splitting datasets.

### 5. Running Tests
ReviewMind includes a lightweight test suite using Python's built-in `unittest` framework that runs independently of large dataset files:

```bash
python -m unittest discover tests
```

Ensure all tests pass before submitting your Pull Request. If you introduce new features, add corresponding tests in the `tests/` directory.

### 6. Commit Guidelines
Write clear, imperative commit messages:

- `feat: add sublinear tf-idf scaling option`
- `fix: resolve encoding warning in external dataset loader`
- `docs: update installation instructions for Linux`
- `test: add unit test for sentiment mapping edge cases`

### 7. Submitting a Pull Request
1. Push your branch to your fork:
   ```bash
   git push origin feature/improved-tokenization
   ```
2. Open a Pull Request against the `main` branch of `RYAD-BENYAKOUB/ReviewMind`.
3. Fill in the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) with details on your changes and testing.

---

## Attribution & Data Safety

- **Never commit raw dataset files (`.csv`) or serialized models (`.joblib`)**: these are excluded by `.gitignore`.
- **Never commit confidential information**: ensure no API keys, private paths, or personal tokens are present in your changes.
- All contributions are licensed under the project's [MIT License](LICENSE).

