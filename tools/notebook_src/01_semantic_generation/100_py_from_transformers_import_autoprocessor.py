from transformers import AutoProcessor

# transformers v5 renamed the model classes' import path in places; fall back to the generic
# image-text-to-text auto class, which resolves Qwen2.5-VL from the checkpoint config.
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _VLModel

torch.manual_seed(SEED)

_kw = dict(attn_implementation="sdpa",     # Turing: no flash-attn
           device_map="auto")              # split the 7B across both T4s
if LOAD_4BIT:
    # Optional single-card path. Needs bitsandbytes, therefore needs internet.
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
# Kaggle model mounts are sometimes missing preprocessor_config.json, in which case
# AutoProcessor cannot resolve the image processor and raises
#   "Unrecognized image processor in <path>"
# even though the weights are fine. The cascade below tries the normal routes first, then
# builds the processor from its parts.
print("files in model dir:", sorted(os.listdir(MODEL_PATH)))


def _patched_config_dir(path, dst="/kaggle/working/_qwen_processor"):
    """Copy only the small config/tokenizer files to a writable dir and inject the keys
    transformers v5 needs.

    This checkpoint was saved by transformers 4.x, whose preprocessor_config.json has no
    `image_processor_type`. v5 no longer falls back to config.json's `model_type`, so
    AutoProcessor cannot resolve the class. Adding the key by hand fixes it. The mount is
    read-only, hence the copy -- weights are NOT copied, only json/txt (a few MB).
    """
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(path):
        if f.endswith((".json", ".txt")) and f != "model.safetensors.index.json":
            shutil.copy2(os.path.join(path, f), os.path.join(dst, f))

    pc = os.path.join(dst, "preprocessor_config.json")
    cfg = json.load(open(pc)) if os.path.exists(pc) else {}
    cfg.setdefault("image_processor_type", "Qwen2_5_VLImageProcessor")
    cfg.setdefault("processor_class", "Qwen2_5_VLProcessor")
    cfg.setdefault("video_processor_type", "Qwen2_5_VLVideoProcessor")
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
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor
        try:
            from transformers import Qwen2_5_VLImageProcessor as _IP
        except ImportError:
            from transformers import Qwen2VLImageProcessor as _IP
        # v5 requires a video processor even for image-only use.
        _VP = None
        for _n in ("Qwen2_5_VLVideoProcessor", "Qwen2VLVideoProcessor"):
            try:
                _VP = getattr(__import__("transformers", fromlist=[_n]), _n)
                break
            except Exception:
                pass
        tok  = AutoTokenizer.from_pretrained(path)
        ip   = _IP(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        kw   = dict(image_processor=ip, tokenizer=tok)
        if _VP is not None:
            kw["video_processor"] = _VP()
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

# A hand-built processor has no chat template; recover it from the mount or the tokenizer,
# otherwise apply_chat_template later would fail.
if not getattr(processor, "chat_template", None):
    _ct = None
    _ctf = os.path.join(MODEL_PATH, "chat_template.json")
    if os.path.exists(_ctf):
        _ct = json.load(open(_ctf)).get("chat_template")
    if _ct is None:
        _ct = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    if _ct:
        processor.chat_template = _ct
        print("  chat template recovered")
    else:
        print("  WARNING: no chat template found -- apply_chat_template may fail")

LOAD_SECONDS = time.time() - _t0
MODEL_REV = os.path.basename(MODEL_PATH.rstrip("/"))

print(f"loaded in {LOAD_SECONDS:.0f}s | 4bit={LOAD_4BIT}")
print("device map:", getattr(model, "hf_device_map", "single device"))


def _set_pixel_budget(max_pixels):
    """Swap the image processor's area budget for one call, returning the previous value.

    Categories with extreme aspect ratios (sheet_metal is 4:1) lose too much detail under
    the shared budget, because the cap is on total area and shrinks both axes.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return None
    prev = getattr(ip, "max_pixels", None)
    try:
        ip.max_pixels = max_pixels
        # Some versions mirror the caps inside a `size` dict; keep it consistent.
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = max_pixels
    except Exception:
        return None
    return prev


def vlm(img_path, system, question, max_new_tokens=200, temperature=0.0, max_pixels=None):
    """One VLM call: image + question -> (raw text reply, generate seconds).

    The timer starts only AFTER the image has been read, preprocessed and moved to the GPU,
    so the reported figure is the time Qwen spends looking at the image and answering --
    not disk I/O or tokenisation.

    temperature=0 is greedy and deterministic; >0 samples, which is how later images in a
    category are pushed away from the first.
    """
    prev_budget = _set_pixel_budget(max_pixels) if max_pixels else None
    img  = Image.open(img_path).convert("RGB")
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]}]
    text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    if prev_budget is not None:
        _set_pixel_budget(prev_budget)          # restore before the next category
    kw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.9)

    if torch.cuda.is_available():
        torch.cuda.synchronize()          # do not time work still queued from before
    t0 = time.time()                      # <-- clock starts: image is now with Qwen
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()          # generate is async; wait before stopping the clock
    secs = time.time() - t0

    # Strip the prompt tokens; keep only what the model generated.
    reply = processor.decode(out[0][inputs.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
    return reply, secs


def grab_json(raw):
    """Pull the first JSON object out of a reply, tolerating stray prose or code fences."""
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(0))
