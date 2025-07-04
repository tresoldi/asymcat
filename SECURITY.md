# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of ASymCat:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take security seriously and appreciate your efforts to responsibly disclose vulnerabilities.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by:

1. **Email**: Send an email to [tiago.tresoldi@lingfil.uu.se](mailto:tiago.tresoldi@lingfil.uu.se) with the subject line "SECURITY: [Brief description]"
2. **GitHub Security Advisories**: Use the [GitHub Security Advisory](https://github.com/tresoldi/asymcat/security/advisories) feature

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Status Updates**: We will keep you informed of our progress every 7 days until resolution
- **Resolution**: We aim to resolve security issues within 30 days when possible

### Disclosure Policy

- We follow a coordinated disclosure approach
- We will work with you to determine an appropriate disclosure timeline
- We will publicly acknowledge your contribution (unless you prefer to remain anonymous)
- We may provide a security advisory through GitHub Security Advisories

## Security Best Practices

### For Users

- Always use the latest stable version of ASymCat
- Keep your Python environment and dependencies up to date
- Validate and sanitize any user input before processing with ASymCat
- Be cautious when processing data from untrusted sources

### For Contributors

- Follow secure coding practices
- Use static analysis tools (bandit, safety) before submitting code
- Never commit secrets, API keys, or passwords
- Use dependency scanning tools to check for vulnerable dependencies
- Write security tests for security-sensitive functionality

## Automated Security Scanning

This project uses several automated security scanning tools:

- **Bandit**: Python security linter for common security issues
- **Safety**: Checks dependencies for known security vulnerabilities
- **Pip-audit**: Audits Python packages for known vulnerabilities
- **CodeQL**: Semantic code analysis engine for finding security vulnerabilities
- **Dependency Review**: GitHub's dependency vulnerability scanning

These tools run automatically on:
- Every pull request
- Daily scheduled scans
- Every push to the main branch

## Security Configuration

### Pre-commit Hooks

Security checks are integrated into our pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run security checks
pre-commit run bandit --all-files
```

### Manual Security Scanning

You can run security scans manually:

```bash
# Install security dependencies
pip install "asymcat[security]"

# Run Bandit
bandit -r asymcat/

# Run Safety
safety check

# Run Pip Audit
pip-audit
```

## Dependencies Security

We regularly review and update our dependencies to address security vulnerabilities. Our dependency management follows these principles:

- Pin dependency versions in `pyproject.toml`
- Regularly update dependencies to their latest secure versions
- Monitor security advisories for all dependencies
- Use automated tools to detect vulnerable dependencies

## Contact

For any questions about this security policy, please contact:

- **Email**: tiago.tresoldi@lingfil.uu.se
- **GitHub**: [@tresoldi](https://github.com/tresoldi)

---

**Note**: This security policy is subject to change. Please check back regularly for updates.