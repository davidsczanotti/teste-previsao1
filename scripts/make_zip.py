# scripts/make_zip.sh
import os, zipfile

EXCLUDES = {
    "reports", "lightning_logs", "mlruns",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints",
    "build", "dist",
}

INCLUDE_TOP = {"src", "configs", "pyproject.toml", "poetry.lock", "README.md", "AGENTS.md", "docs", "scripts"}

def should_skip(path):
    parts = path.split(os.sep)
    for p in parts:
        if p in EXCLUDES:
            return True
    # ignora egg-info e diretórios de build
    if any(x in path for x in (".egg-info",)):
        return True
    return False

def main():
    """Cria um arquivo zip do projeto, excluindo arquivos desnecessários."""
    zip_name = "release.zip"
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE_TOP:
            if not os.path.exists(item):
                continue
            if os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    # filtra dirs in-place
                    dirs[:] = [d for d in dirs if not should_skip(os.path.join(root, d))]
                    for f in files:
                        full = os.path.join(root, f)
                        if should_skip(full):
                            continue
                        zf.write(full)
            else:
                zf.write(item)
    
    print(f"=> criado {zip_name}")

if __name__ == "__main__":
    main()
