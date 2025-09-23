import shutil, os, glob

TO_WIPE = [
    "reports",
    "lightning_logs",
    "mlruns",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "*.egg-info",
]


def rm(p):
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    elif os.path.isfile(p):
        try:
            os.remove(p)
        except:
            pass


for pat in TO_WIPE:
    for path in glob.glob(pat):
        print("removendo:", path)
        rm(path)
print("ok!")
