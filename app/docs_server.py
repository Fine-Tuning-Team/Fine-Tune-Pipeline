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
    print("🔍 Checking documentation dependencies...")
    
    # First, try to check if mkdocs is already available
    try:
        subprocess.run(["mkdocs", "--version"], check=True, capture_output=True)
        print("✅ MkDocs is already available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 MkDocs not found, need to install dependencies")
    
    # Try uv first
    if check_uv_installed():
        print("🔄 Using uv to install dependencies...")
        if run_command("uv sync --extra docs", "Installing documentation dependencies with uv"):
            return True
        else:
            print("⚠️  uv sync failed, trying pip fallback...")
    else:
        print("⚠️  uv not found, using pip fallback...")
    
    # Fallback to pip
    print("🔄 Using pip to install dependencies...")
    deps = [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.0.0", 
        "mkdocstrings[python]>=0.24.0",
        "mkdocs-mermaid2-plugin>=1.1.0",
        "pymdown-extensions"
    ]
    
    pip_cmd = f"pip install {' '.join(deps)}"
    if run_command(pip_cmd, "Installing documentation dependencies with pip"):
        return True
    
    print("❌ Failed to install dependencies with both uv and pip")
    print("💡 You can manually install dependencies:")
    print("   pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin pymdown-extensions")
    return False

def serve_docs():
    """Serve documentation locally."""
    print("🚀 Starting documentation server...")
    print("📖 Documentation will be available at: http://127.0.0.1:8000")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Try uv first, then fallback to direct mkdocs
    commands_to_try = [
        ["uv", "run", "mkdocs", "serve"],
        ["mkdocs", "serve"]
    ]
    
    for cmd in commands_to_try:
        try:
            subprocess.run(cmd, check=True)
            break
        except KeyboardInterrupt:
            print("\n👋 Documentation server stopped")
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if cmd == commands_to_try[-1]:  # Last command failed
                print(f"❌ Failed to start documentation server: {e}")
            else:
                continue  # Try next command

def build_docs():
    """Build static documentation."""
    # Try uv first, then fallback to direct mkdocs
    commands_to_try = [
        "uv run mkdocs build",
        "mkdocs build"
    ]
    
    for cmd in commands_to_try:
        if run_command(cmd, "Building static documentation"):
            print("📁 Static documentation built in ./site/ directory")
            return True
    
    return False

def main():
    """Main function."""
    print("📚 Fine-Tune Pipeline Documentation Setup")
    print("=" * 50)
    
    # Check if we're in the right directory (look for mkdocs.yml in parent directory)
    project_root = Path(__file__).parent.parent
    mkdocs_file = project_root / "mkdocs.yml"
    
    if not mkdocs_file.exists():
        print("❌ mkdocs.yml not found. Please ensure this script is in the app/ directory of the project.")
        print(f"Looking for: {mkdocs_file}")
        sys.exit(1)
    
    # Change to project root directory for mkdocs commands
    os.chdir(project_root)
    print(f"📁 Working directory: {project_root}")
    
    # Ask if user wants to skip dependency check
    print("\nDo you want to check/install documentation dependencies?")
    print("(Choose 'n' if you already have mkdocs installed)")
    install_deps = input("Install dependencies? (y/n): ").strip().lower()
    
    if install_deps in ['y', 'yes', '']:
        # Install dependencies
        if not install_docs_dependencies():
            print("\n⚠️  Dependency installation failed, but you can still try to continue...")
            continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
            if continue_anyway not in ['y', 'yes']:
                sys.exit(1)
    else:
        print("⏭️  Skipping dependency installation")
    
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
