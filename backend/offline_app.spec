# Build with:
#   python -m PyInstaller --clean --noconfirm offline_app.spec
#
# The generated folder is:
#   backend/offline_dist/EasyDefect_Offline/
#
# The executable expects a sibling model/ folder containing:
#   encoder.xml, encoder.bin, autoencoder.xml, autoencoder.bin, threshold.joblib

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

openvino_datas = collect_data_files("openvino")
openvino_binaries = collect_dynamic_libs("openvino")

a = Analysis(
    ["offline_app.py"],
    pathex=[],
    binaries=openvino_binaries,
    datas=openvino_datas,
    hiddenimports=[
        "openvino",
        "openvino.runtime",
        "openvino._pyopenvino",
        "flask",
        "werkzeug",
        "joblib",
        "numpy",
        "PIL",
        "PIL.Image",
        "utils.offline_bundle_html",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cv2",
        "IPython",
        "keras",
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "tensorflow",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EasyDefect_Offline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EasyDefect_Offline",
)
