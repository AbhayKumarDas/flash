from IPython.display import FileLink, display

ZIP_PATH = os.path.join(OUT_ROOT, "Final Synthetic Images.zip")
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(SYN_DIR):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, OUT_ROOT))

print(f"{ZIP_PATH}  ({os.path.getsize(ZIP_PATH)/1e6:.1f} MB, {n_img} images)")
display(FileLink(ZIP_PATH))
