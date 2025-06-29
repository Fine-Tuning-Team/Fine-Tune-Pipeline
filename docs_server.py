#!/usr/bin/env python3
"""
Documentation setup script for Fine-Tune Pipeline.
This script helps set up and serve the documentation locally.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_docs_dependencies():
    """Install documentation dependencies."""
    if not check_uv_installed():
        print("❌ uv is not installed. Please install uv first:")
        print("   https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    # Install docs dependencies
    if not run_command("uv sync --extra docs", "Installing documentation dependencies"):
        return False
    
    return True

def serve_docs():
    """Serve documentation locally."""
    print("🚀 Starting documentation server...")
    print("📖 Documentation will be available at: http://127.0.0.1:8000")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(["uv", "run", "mkdocs", "serve"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Documentation server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start documentation server: {e}")

def build_docs():
    """Build static documentation."""
    if run_command("uv run mkdocs build", "Building static documentation"):
        print("📁 Static documentation built in ./site/ directory")
        return True
    return False

def main():
    """Main function."""
    print("📚 Fine-Tune Pipeline Documentation Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("mkdocs.yml").exists():
        print("❌ mkdocs.yml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install dependencies
    # if not install_docs_dependencies():
    #     sys.exit(1)
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. 🌐 Serve documentation locally (development)")
    print("2. 🏗️  Build static documentation")
    print("3. 📋 Both - build and serve")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        serve_docs()
    elif choice == "2":
        build_docs()
    elif choice == "3":
        if build_docs():
            serve_docs()
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
