import torch, re
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
except ImportError:
    # transformers v5 moved the class; the generic auto class resolves it from the config.
    from transformers import AutoModelForImageTextToText as _VLModel


def find_model():
    """Use MODEL_ID if it exists on disk, else search /kaggle/input for the weights."""
    if MODEL_ID and os.path.isdir(MODEL_ID):
        return MODEL_ID
    if MODEL_ID:
        print(f"  MODEL_ID not on disk ({MODEL_ID}) -- searching instead")
    hits = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if not glob.glob(os.path.join(d, "*.safetensors")):
            continue                                   # config without weights -> skip
        try:
            arch = json.load(open(cfg)).get("architectures", [""])[0]
        except Exception:
            arch = ""
        if "Qwen2_5_VL" in arch or "qwen" in d.lower():
            hits.append((d, arch))
    if not hits:
        raise FileNotFoundError("no mounted Qwen model under /kaggle/input -- attach it, "
                                "or set MODEL_ID to a hub id")
    hits.sort(key=lambda t: t[0])
    for d, a in hits:
        print("  model candidate:", d, "|", a)
    return hits[-1][0]


MODEL_PATH = find_model()
print("model:", MODEL_PATH)

torch.manual_seed(SEED)
_kw = dict(attn_implementation="sdpa",     # Turing: no flash-attn
           device_map="auto")              # split the 7B across both T4s
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    _kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

_t0 = time.time()
# v5 renamed torch_dtype -> dtype. Try the new name, fall back to the old one.
try:
    model = _VLModel.from_pretrained(MODEL_PATH, dtype=torch.float16, **_kw).eval()
except TypeError:
    model = _VLModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, **_kw).eval()


# ---------------------------------------------------------------- processor
# Kaggle model mounts predate transformers v5. v5 resolves the image processor from
# `image_processor_type` in preprocessor_config.json and NO LONGER falls back to config.json's
# `model_type`, so a 4.x checkpoint raises "Unrecognized image processor" even though the
# weights loaded fine. The cascade below tries the normal routes, then patches a copy of the
# small config files, then finally builds the processor FROM ITS PARTS -- which bypasses
# config-driven class resolution altogether and is the route that works when the rest do not.
print("files in model dir:", sorted(os.listdir(MODEL_PATH)))
_pcf = os.path.join(MODEL_PATH, "preprocessor_config.json")
if os.path.exists(_pcf):
    try:
        print("preprocessor_config.json keys:", sorted(json.load(open(_pcf)).keys()))
    except Exception as e:
        print("preprocessor_config.json unreadable:", e)
else:
    print("preprocessor_config.json: ABSENT")


def _image_processor_cls():
    """The image-processor class that actually exists in THIS transformers build.

    Hardcoding "Qwen2_5_VLImageProcessor" into a patched config is wrong when the installed
    version only ships Qwen2VLImageProcessor -- Qwen2.5-VL reuses Qwen2-VL's image processor in
    several releases. Resolve the name at runtime rather than guessing it.
    """
    import transformers
    for n in ("Qwen2_5_VLImageProcessor", "Qwen2VLImageProcessor"):
        if hasattr(transformers, n):
            return n, getattr(transformers, n)
    return None, None


def _video_processor_cls():
    """v5 wants a video processor even for image-only use. Absent in 4.x, which is fine."""
    import transformers
    for n in ("Qwen2_5_VLVideoProcessor", "Qwen2VLVideoProcessor"):
        if hasattr(transformers, n):
            return n, getattr(transformers, n)
    return None, None


IP_NAME, IP_CLS = _image_processor_cls()
VP_NAME, VP_CLS = _video_processor_cls()
print(f"image processor class: {IP_NAME or 'NONE FOUND'} | video: {VP_NAME or 'none'}")


def _patched_config_dir(path, dst="/kaggle/working/_qwen_processor"):
    """Copy only the small config/tokenizer files to a writable dir and inject the v5 keys.

    The mount is read-only, hence the copy -- weights are NOT copied, only json/txt (a few MB).
    The destination is cleared first: a half-written dir left by an earlier failed attempt
    would otherwise be picked up and fail again for a different reason.
    """
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(path):
        if f.endswith((".json", ".txt")) and f != "model.safetensors.index.json":
            shutil.copy2(os.path.join(path, f), os.path.join(dst, f))

    pc = os.path.join(dst, "preprocessor_config.json")
    try:
        cfg = json.load(open(pc)) if os.path.exists(pc) else {}
    except Exception:
        cfg = {}
    if IP_NAME:
        cfg["image_processor_type"] = IP_NAME    # set, not setdefault: overwrite a stale value
    cfg.setdefault("processor_class", "Qwen2_5_VLProcessor")
    if VP_NAME:
        cfg.setdefault("video_processor_type", VP_NAME)
    json.dump(cfg, open(pc, "w"), indent=2)
    return dst


def load_processor(path):
    errors = []

    for kw in (dict(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, use_fast=False),
               dict(use_fast=False)):                    # 1) straight AutoProcessor
        try:
            return AutoProcessor.from_pretrained(path, **kw)
        except Exception as e:
            errors.append(f"AutoProcessor({', '.join(kw)}): {type(e).__name__}: {e}")

    try:                                                 # 2) patched config dir
        pdir = _patched_config_dir(path)
        proc = AutoProcessor.from_pretrained(pdir, min_pixels=MIN_PIXELS,
                                             max_pixels=MAX_PIXELS, use_fast=False)
        print(f"  loaded via patched config dir ({pdir}) -- "
              "checkpoint predates the v5 image_processor_type key")
        return proc
    except Exception as e:
        errors.append(f"patched dir: {type(e).__name__}: {e}")

    try:                                                 # 3) concrete class, patched dir
        from transformers import Qwen2_5_VLProcessor
        return Qwen2_5_VLProcessor.from_pretrained(_patched_config_dir(path))
    except Exception as e:
        errors.append(f"Qwen2_5_VLProcessor: {type(e).__name__}: {e}")

    try:                                                 # 4) assemble from parts
        # The route that survives when the config cannot be resolved at all: every class is
        # instantiated directly, so nothing ever reads image_processor_type.
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor
        if IP_CLS is None:
            raise ImportError("no Qwen image processor class in this transformers build")
        tok = AutoTokenizer.from_pretrained(path)
        try:
            ip = IP_CLS(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        except TypeError:
            ip = IP_CLS()                                # older signature; caps applied below
        kw = dict(image_processor=ip, tokenizer=tok)
        if VP_CLS is not None:
            try:
                kw["video_processor"] = VP_CLS()
            except Exception:
                pass                                     # image-only use; not fatal
        print("  built processor from tokenizer + image processor")
        return Qwen2_5_VLProcessor(**kw)
    except Exception as e:
        errors.append(f"manual build: {type(e).__name__}: {e}")

    raise RuntimeError("could not build a processor:\n  " + "\n  ".join(errors))


processor = load_processor(MODEL_PATH)

# Apply the pixel caps if they did not go in through the constructor.
_ip = getattr(processor, "image_processor", None)
if _ip is not None:
    for _k, _v in (("min_pixels", MIN_PIXELS), ("max_pixels", MAX_PIXELS)):
        try:
            setattr(_ip, _k, _v)
        except Exception:
            pass

# A hand-built processor has NO chat template, and vlm() calls apply_chat_template in the very
# next cell -- so recover it from the mount or the tokenizer before anything else runs.
if not getattr(processor, "chat_template", None):
    _ct = None
    _ctf = os.path.join(MODEL_PATH, "chat_template.json")
    if os.path.exists(_ctf):
        try:
            _ct = json.load(open(_ctf)).get("chat_template")
        except Exception:
            _ct = None
    if _ct is None:
        _ct = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    if _ct:
        processor.chat_template = _ct
        print("  chat template recovered")
    else:
        print("  WARNING: no chat template found -- apply_chat_template may fail")

LOAD_SECONDS = time.time() - _t0
# Reported separately everywhere below. It is paid once per session, not per crop, so folding
# it into a per-crop figure would misstate Stage 3's cost -- same convention as Stage 1.
print(f"loaded in {LOAD_SECONDS:.0f}s | 4bit={LOAD_4BIT}")
print("device map:", getattr(model, "hf_device_map", "single device"))
