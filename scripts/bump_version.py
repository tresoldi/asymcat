#!/usr/bin/env python3
"""
Version bumping script for ASymCat.

This script updates the version in both pyproject.toml and asymcat/__init__.py
to ensure consistency across the project.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

try:
    import tomllib
except ImportError:
    import toml as tomllib_fallback


def get_current_version() -> str:
    """Get the current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found. Run this script from the project root.")
    
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except NameError:
        # Fallback for Python < 3.11
        with open(pyproject_path, "r") as f:
            data = tomllib_fallback.load(f)
    
    return data["project"]["version"]


def parse_version(version: str) -> Tuple[int, int, int, str]:
    """Parse a version string into components."""
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:[-.]?(alpha|beta|rc|dev)\.?(\d+)?)?$"
    match = re.match(pattern, version)
    
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    
    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
    prerelease = match.group(4) or ""
    
    return major, minor, patch, prerelease


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type."""
    major, minor, patch, prerelease = parse_version(current)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "alpha":
        return f"{major}.{minor}.{patch + 1}alpha1" if not prerelease else current
    elif bump_type == "beta":
        return f"{major}.{minor}.{patch + 1}beta1" if not prerelease else current
    elif bump_type == "rc":
        return f"{major}.{minor}.{patch + 1}rc1" if not prerelease else current
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_pyproject_toml(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Replace version line
    pattern = r'^version = "[^"]*"'
    replacement = f'version = "{new_version}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    pyproject_path.write_text(new_content)
    print(f"Updated pyproject.toml: version = \"{new_version}\"")


def update_init_file(new_version: str) -> None:
    """Update version in asymcat/__init__.py."""
    init_path = Path("asymcat/__init__.py")
    content = init_path.read_text()
    
    # Replace version line
    pattern = r'^__version__ = "[^"]*"'
    replacement = f'__version__ = "{new_version}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    init_path.write_text(new_content)
    print(f"Updated asymcat/__init__.py: __version__ = \"{new_version}\"")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Bump version for ASymCat")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch", "alpha", "beta", "rc"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--new-version",
        help="Specific version to set (overrides bump_type)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    
    args = parser.parse_args()
    
    try:
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        
        if args.new_version:
            new_version = args.new_version
            # Validate the new version format
            parse_version(new_version)
        else:
            new_version = bump_version(current_version, args.bump_type)
        
        print(f"New version: {new_version}")
        
        if args.dry_run:
            print("\n[DRY RUN] Would update:")
            print(f"  - pyproject.toml: version = \"{new_version}\"")
            print(f"  - asymcat/__init__.py: __version__ = \"{new_version}\"")
        else:
            update_pyproject_toml(new_version)
            update_init_file(new_version)
            
            print(f"\n✅ Version bumped from {current_version} to {new_version}")
            print("\nNext steps:")
            print(f"1. git add pyproject.toml asymcat/__init__.py")
            print(f"2. git commit -m \"Bump version to {new_version}\"")
            print(f"3. git tag v{new_version}")
            print(f"4. git push origin master --tags")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()