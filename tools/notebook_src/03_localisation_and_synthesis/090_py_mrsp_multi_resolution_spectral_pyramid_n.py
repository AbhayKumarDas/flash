# --- MRSP: multi-resolution spectral pyramid noise (from flash_part1) ---------------
def generate_mrsp(h, w, scale=8, device="cpu", seed=None, alpha=1.5,
                  levels=OCTAVES_FBM, persistence=PERSISTENCE):
    if seed is not None:
        torch.manual_seed(int(seed))
    device = torch.device(device)
    field = torch.zeros(h, w, device=device, dtype=torch.float32)
    weight_sq_sum = 0.0
    for level in range(levels):
        factor = 2 ** (levels - 1 - level)
        h_l = max(8, h // factor); w_l = max(8, w // factor)
        fy = torch.fft.fftfreq(h_l, device=device).reshape(-1, 1)
        fx = torch.fft.rfftfreq(w_l, device=device).reshape(1, -1)
        f = torch.sqrt(fy * fy + fx * fx) * (1.0 + math.log2(max(1, int(scale))))
        amp = torch.where(f > 1e-6, f.pow(-alpha), torch.zeros_like(f))
        phase = 2 * math.pi * torch.rand(h_l, w_l // 2 + 1, device=device)
        spec = amp * torch.complex(torch.cos(phase), torch.sin(phase))
        noise_lr = torch.fft.irfft2(spec, s=(h_l, w_l))
        if (h_l, w_l) != (h, w):
            noise = F.interpolate(noise_lr.unsqueeze(0).unsqueeze(0), size=(h, w),
                                  mode="bilinear", align_corners=False).squeeze()
        else:
            noise = noise_lr
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)
        w_oct = persistence ** level
        field = field + w_oct * noise
        weight_sq_sum += w_oct ** 2
    field = field / math.sqrt(weight_sq_sum + 1e-8)
    return (field - field.mean()) / (field.std() + 1e-8)


# --- OBS: object-aware foreground (from flash_part1) --------------------------------
def _fg_robust(image, use_cues=True):
    img = (np.clip(image.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]], 0)
    bg = np.median(border, 0)
    dist = np.sqrt(((lab - bg) ** 2).sum(2))
    du8 = (255 * dist / (dist.max() + 1e-8)).astype(np.uint8)
    _, cm = cv2.threshold(du8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cm > 0
    if use_cues:
        gray = img.mean(2).astype(np.float32) / 255.0
        k = max(15, round(min(img.shape[:2]) / 64))
        mean = ndimage.uniform_filter(gray, k)
        sq = ndimage.uniform_filter(gray * gray, k)
        std = np.sqrt(np.clip(sq - mean * mean, 0, None))
        su8 = (255 * std / (std.max() + 1e-8)).astype(np.uint8)
        _, tm = cv2.threshold(su8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = m | (tm > 0)
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m)
    if n:
        sz = ndimage.sum(np.ones_like(lbl, dtype=np.float32), lbl, range(1, n + 1))
        keep = np.where(sz >= 0.0015 * m.size)[0] + 1
        if len(keep) == 0:
            keep = [int(sz.argmax()) + 1]
        m = np.isin(lbl, keep)
    return m.astype(np.float32)


def region_for(image):
    """Algorithm 2 step 11 -- the degeneracy guard."""
    region = _fg_robust(image, use_cues=True); cov = region.mean()
    if cov < 0.03 or cov > 0.99:
        region = np.ones_like(region)
    return torch.from_numpy(region).float()


def coverage_mask(noise, region, target_coverage):
    region = region.to(noise.device)
    values = noise[region > 0]
    if values.numel() == 0:
        return torch.zeros_like(noise)
    thr = torch.quantile(values, 1.0 - target_coverage)
    return ((noise > thr) & (region > 0)).float()


def mrsp_blob_stats(mask, min_blob_px=MIN_BLOB_PX):
    """Area-weighted characteristic blob side + largest-blob centroid."""
    lbl, n = ndimage.label(mask.numpy() > 0.5)
    if n == 0:
        return 0.0, None
    areas = ndimage.sum(np.ones_like(lbl, dtype=np.float32), lbl, range(1, n + 1))
    big = areas[areas >= min_blob_px]
    if big.size == 0:
        big = areas
    char_side = math.sqrt(float((big ** 2).sum() / big.sum()))
    i = int(areas.argmax()) + 1
    ys, xs = np.where(lbl == i)
    return char_side, (int(ys.mean()), int(xs.mean()))


def keep_largest_blob(mask):
    lbl, k = ndimage.label(mask.numpy() > 0.5)
    if k <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl, np.float32), lbl, range(1, k + 1))
    return torch.from_numpy((lbl == int(sizes.argmax()) + 1).astype(np.float32))


MRSP_DEVICE = "cpu"          # replaced by the measurement in section 4

def mrsp_mask_for(img, seed=SEED, region=None, coverage=None):
    _, H, W = img.shape
    if region is None:
        region = region_for(img)
    noise = generate_mrsp(H, W, scale=AD2_NOISE_SCALE, device=MRSP_DEVICE, seed=seed,
                          alpha=MRSP_ALPHA).cpu()
    mask = coverage_mask(noise, region, AD2_COVERAGE if coverage is None else coverage)
    if SINGLE_BLOB:
        mask = keep_largest_blob(mask)
    return region, mask


# OBS is deterministic per host and is recomputed for every seed unless cached. At 1024px it is
# ~0.2s of morphology; three seeds over 24 hosts is 72 recomputations of the same answer.
_HOST_CACHE = {}
def host_and_region(path):
    key = (path, WORK_SIZE)
    if key not in _HOST_CACHE:
        img = load_image(path)
        _HOST_CACHE[key] = (img, region_for(img))
    img, reg = _HOST_CACHE[key]
    return img.clone(), reg

print("MRSP + OBS ready.")
