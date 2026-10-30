# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

We take the security of ReviewMind seriously. If you discover a security vulnerability, please report it privately rather than opening a public issue.

- **Security Contact**: Contact the maintainer at `ryadbenyakoub@gmail.com`.
- Please include:
  - A description of the vulnerability and its potential impact.
  - Clear steps or a proof-of-concept to reproduce the issue.
  - Your proposed mitigation, if any.
- You will receive an acknowledgment within 48 hours, followed by regular updates until a patch is issued.

---

## Security Considerations & Best Practices

### 1. Serialized Model Files (`.joblib` / `.pkl`)
ReviewMind saves and loads trained pipelines using `joblib`.
> **Warning**: Loading serialized models with `joblib` or `pickle` can execute arbitrary Python code.
> - **Only load `.joblib` files** that you trained locally or that come from officially verified releases.
> - **Never load untrusted `.joblib` files** obtained from unknown third parties.

### 2. Environment Variables & Credentials
- ReviewMind does not require API keys, database credentials, or access tokens to run.
- Do not add API keys or credentials to the codebase or commit `.env` files.
- The `.gitignore` file is configured to exclude all `.env` files except `.env.example`.

### 3. Data Privacy
ReviewMind is designed to process public hotel reviews. Ensure that any customized corpora fed into the pipeline do not contain Personally Identifiable Information (PII) or confidential business records.

