# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('icon/eye.png', 'icon'),
        ('models/yolov8n.onnx', 'models'),
        ('models/yolov8m.onnx', 'models'),
        ('models/yolov8l.onnx', 'models'),
        ('models/yolov8x.onnx', 'models'),
        ('onnxmodules/__init__.py', 'onnxmodules'),
        ('onnxmodules/utils.py', 'onnxmodules'),
        ('onnxmodules/yolov8onnx.py', 'onnxmodules'),
        ('ui/new_gui.py', 'ui'),
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages\onnxruntime\capi\*.dll", 'onnxruntime\capi'),
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages\onnxruntime\capi\*.pyd", 'onnxruntime\capi'),
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages\numpy", 'numpy'),  # NumPy ekleniyor
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages\pip", 'pip'),  # Pip ekleniyor
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages\mkl_random", 'mkl_random'),  # MKL random ekleniyor
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages\mkl_fft", 'mkl_fft'),  # MKL FFT ekleniyor
        (r"C:\Users\aakyilm\AppData\Local\anaconda3\envs\onnx_gpu_pdetection\Lib\site-packages", 'six'),  # Six ekleniyor
    ],

    hiddenimports=[
        'cv2',
        'PyQt5',
        'sqlalchemy',
        'onnxruntime',
        'pyodbc',
        'importlib_metadata',
        'MySQLdb',
        'psycopg2',
        'sip',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='icon/eye.png'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main'
)
