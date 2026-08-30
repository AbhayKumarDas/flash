from IPython.display import FileLink, display

for name, src in (("defect_bank", BANK_DIR),
                  ("Defect Masks and Anomalies", PAIRS_DIR)):
    zp = os.path.join(OUT_ROOT, f"{name}.zip")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, OUT_ROOT))
    print(f"{zp}  ({os.path.getsize(zp)/1e6:.1f} MB)")
    display(FileLink(zp))
