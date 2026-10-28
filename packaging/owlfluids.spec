# PyInstaller spec for the Windows build: `pyinstaller packaging/owlfluids.spec`
# (packaging/build.ps1 runs the whole build). Produces dist/OwlFluids/ with
# owlfluids.exe (command line) and owlfluids-app.exe (desktop app).
#
# OwlFluids itself is collected as .py source files rather than into the archive:
# numba's on-disk kernel cache needs the source files to exist, and so does the
# cache-key hash in owlfluids/__init__.py.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []
for pkg in ("CoolProp", "pyqtgraph", "synnax"):
    try:
        d, b, h = collect_all(pkg)
    except Exception as e:  # synnax is optional in a local build
        print(f"spec: skipping {pkg}: {e}")
        continue
    datas += d
    binaries += b
    hiddenimports += h

# the desktop app's bundled fonts (Lato, Roboto Mono, Barlow Semi Condensed; all OFL)
datas.append((str(ROOT / "owlfluids" / "app" / "fonts"), "owlfluids/app/fonts"))

bundled = ROOT / "owlfluids" / "fluids" / "bundled"
if any(bundled.glob("*.npz")):
    datas.append((str(bundled), "owlfluids/fluids/bundled"))
else:
    print("spec: no bundled property tables; run `owlfluids tables --out owlfluids/fluids/bundled` first")

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["owlfluids.app.main", "owlfluids.drivers.synnax", "h5py"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2"],
    module_collection_mode={"owlfluids": "py"},
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = ROOT / "packaging" / "owlfluids.ico"
common = dict(exclude_binaries=True, debug=False, strip=False, upx=False,
              icon=str(icon) if icon.exists() else None)
cli = EXE(pyz, a.scripts, [], name="owlfluids", console=True, **common)
app = EXE(pyz, a.scripts, [], name="owlfluids-app", console=False, **common)

coll = COLLECT(cli, app, a.binaries, a.datas, strip=False, upx=False, name="OwlFluids")
