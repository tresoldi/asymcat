# Release Process

This document describes the release process for ASymCat.

## Automated Release Workflow

ASymCat uses GitHub Actions for automated releases. The process is triggered by creating and pushing a version tag.

### Prerequisites

1. Ensure all tests pass on the main branch
2. Update CHANGELOG.md with the new version changes
3. All version numbers are consistent across files

### Release Steps

#### Option 1: Using the Version Bump Script (Recommended)

```bash
# 1. Bump version (patch/minor/major)
python scripts/bump_version.py patch  # or minor, major

# 2. Review changes
git diff

# 3. Commit and tag
git add pyproject.toml asymcat/__init__.py CHANGELOG.md
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin master --tags
```

#### Option 2: Manual Version Update

```bash
# 1. Update version in pyproject.toml and asymcat/__init__.py
# Ensure both files have the same version number

# 2. Commit and tag
git add pyproject.toml asymcat/__init__.py
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin master --tags
```

#### Option 3: Manual Workflow Trigger

You can also trigger releases manually from the GitHub Actions interface:

1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Enter the version (e.g., v0.3.1)
4. Choose whether to upload to Test PyPI
5. Click "Run workflow"

## Release Types

### Production Release

- Triggered by pushing tags matching `v*` (e.g., `v0.3.0`, `v1.0.0`)
- Publishes to PyPI
- Creates GitHub release
- Runs full test suite and security scans

### Pre-release

- Use version tags with alpha/beta/rc suffixes (e.g., `v0.3.0-alpha1`)
- Automatically marked as pre-release on GitHub
- Can be tested before final release

### Test Release

- Use manual workflow trigger with "Upload to Test PyPI" option
- Publishes to Test PyPI (https://test.pypi.org/)
- Useful for testing the release process

## Workflow Details

The release workflow includes these steps:

1. **Validation**: Checks version consistency across files
2. **Testing**: Runs tests on all supported Python versions
3. **Security**: Runs security scans (Bandit, Safety, pip-audit)
4. **Build**: Creates wheel and source distributions
5. **Publish**: Uploads to PyPI using trusted publishing
6. **Release**: Creates GitHub release with changelog

## Version Numbering

ASymCat follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Pre-release Versions

- `X.Y.Z-alpha.N`: Alpha releases (early development)
- `X.Y.Z-beta.N`: Beta releases (feature complete, testing)
- `X.Y.Z-rc.N`: Release candidates (final testing)

## PyPI Trusted Publishing

The release workflow uses PyPI's trusted publishing feature, which doesn't require storing API tokens as secrets. This is configured for:

- **Repository**: tresoldi/asymcat
- **Workflow**: release.yml
- **Environment**: pypi (production) and test-pypi (testing)

## Manual Release (Fallback)

If automated release fails, you can release manually:

```bash
# 1. Build the package
python -m build

# 2. Upload to PyPI
python -m twine upload dist/*

# 3. Create GitHub release manually
```

## Troubleshooting

### Version Consistency Errors

If the workflow fails due to version inconsistency:

1. Check that `pyproject.toml` and `asymcat/__init__.py` have the same version
2. Ensure the git tag matches the version in files (without 'v' prefix)

### Build Failures

1. Check that all tests pass locally
2. Ensure all dependencies are properly specified
3. Verify the package builds locally: `python -m build`

### PyPI Upload Failures

1. Check if the version already exists on PyPI
2. Verify trusted publishing is configured correctly
3. Check for any PyPI-specific requirements (README format, etc.)

## Post-Release

After a successful release:

1. Update the documentation if needed
2. Announce the release (if significant)
3. Monitor for any issues or bug reports
4. Plan next release cycle

## Security Considerations

- All releases are automatically scanned for security vulnerabilities
- Dependencies are checked for known CVEs
- No secrets or API keys are stored in the repository
- Trusted publishing provides secure authentication to PyPI

## Testing Releases

Before major releases, consider:

1. Testing with Test PyPI: `pip install -i https://test.pypi.org/simple/ asymcat==X.Y.Z`
2. Creating pre-release versions for beta testing
3. Running extended test suites in different environments
4. Validating documentation and examples