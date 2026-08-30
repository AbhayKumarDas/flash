# ---------------------------------------------------------------- run knobs
SEED          = 0
# Mounted paths. Each is used if it exists; otherwise the notebook falls back to searching
# /kaggle/input, so a re-mount under a different slug does not break the run.
MODEL_ID      = "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/7b-instruct/2"
DATA_ROOT_ID  = "/kaggle/input/datasets/abhaykdas/mvtec-ad2-reducded"
LOAD_4BIT     = False        # False -> fp16 split across both T4s (no internet needed)
MAX_PIXELS    = 1280 * 28 * 28   # visual-token cap: AD2 natives are far bigger than needed
MIN_PIXELS    = 256 * 28 * 28
OUT_DIR       = "/kaggle/working/Local Testing"

DEFECT_SOURCE = "config"     # config = VLM must pick from CONFIG[cat]["defects"]
                             # derive = it may name its own, using the list as examples
N_IMAGES      = 3            # normal images sampled per category; ONE prompt per image

# Shown to the VLM so it knows what kind of image it is looking at. Dataset-level, not
# category-level -- edit this once per dataset.
DATASET_CONTEXT = (
    "High-resolution industrial inspection images from a fixed camera rig under controlled "
    "illumination. Every image in the set shows the same object class under the same "
    "acquisition setting. Real defects in this kind of data are small relative to the frame, "
    "occur anywhere across the full height and width including right at the image borders, "
    "and are often low contrast against the material they sit on."
)

# ---------------------------------------------------------------- per-category config
#   defects : allowed defect list, used only when DEFECT_SOURCE == "config"
#   notes   : category hazards, always passed to the VLM as a hint (e.g. "air bubbles are
#             normal, not defects" stops it proposing bubbles as an anomaly)
# Both fields are optional. Delete a category to skip it; add one to extend to a new dataset.
CONFIG = {
    "can":         {"object": "a printed aluminium soda can lying on a plain background",
                    "families": ["print or coating defect", "scratch or cut", "stain or discolouration",
                                 "deformation"],
                    "defects": ["scratch tearing through the printed label so the bare silver metal underneath shows",
                                "smeared print",
                                "faded patch in the print",
                                "dark ink speck",
                                "dented rim"],
                    "notes": 
"diffuse bright-field front light on a printed soda can. The can rotates between "
                             "shots so print position varies, and the metal throws bright specular highlights -- "
                             "both normal. Real defects here are tiny against busy print"},
    "fabric":      {"object": "a flat sheet of printed woven fabric filling the frame",
                    "families": ["hole or tear", "scratch or cut", "fibre or thread contamination",
                                 "stain or discolouration", "foreign body contamination"],
                    "defects": ["tiny red stain spot",
                                "leftover thread lying on the surface",
                                "loose scrap of fabric on the surface",
                                "hole through the weave",
                                "cut across the weave",
                                "pulled thread"],
                    "notes": 
"diffuse bright-field front light for fabric inspection. The printed pattern itself "
                             "varies between samples, so pattern variation is normal. Real defects are tiny and "
                             "low-contrast against that print"},
    "fruit_jelly": {"object": "a sealed tub of solid set fruit jelly with pieces of fruit suspended inside it",
                    "families": ["foreign body contamination", "fibre or thread contamination",
                                 "stain or discolouration", "mould or rot"],
                    "defects": ["glass shard pressed into the surface",
                                "clear plastic fragment",
                                "dark speck of debris",
                                "small insect on the surface",
                                "patch of mould growing on the jelly",
                                "rotten discoloured fruit piece"],
                    "notes": 
"diffuse bright-field back and front light through semi-transparent jelly. The "
                             "amount, size and layout of the fruit pieces vary between samples and are normal. "
                             "Real defects are low-contrast and partly see-through"},
    "rice":        {"object": "loose white rice grains in bulk, filling the frame",
                    "families": ["foreign body contamination", "fibre or thread contamination",
                                 "stain or discolouration"],
                    "defects": ["clear plastic pellet among the grains",
                                "small stone among the grains",
                                "small piece of glass among the grains",
                                "dark grit particle",
                                "stray fibre across the grains"],
                    "notes": 
"diffuse bright-field front light on rice grains in bulk. Grain layout is random and "
                             "normal. The hard case is contamination that is itself (semi-)transparent, so real "
                             "defects are very low contrast"},
    "sheet_metal": {"object": "several flat strips of machined sheet metal",
                    "families": ["hole or tear", "scratch or cut", "foreign body contamination",
                                 "stain or discolouration"],
                    "defects": ["hole punched through the metal",
                                "deep scratch",
                                "cut in the edge",
                                "short piece of metal wire lying across the surface",
                                "curl of metal swarf on the surface",
                                "dark speck of grit"],
                    "notes": 
"directed dark-field front light on strips of sheet metal. Random specular "
                             "reflections look like defects but are normal. The frame is a wide strip and real "
                             "samples can carry several defects of different sizes",
                    "max_pixels": 2560 * 28 * 28},
    "vial":        {"object": "a single sealed glass vial of clear sparkling liquid with a printed QR label",
                    "families": ["foreign body contamination", "hole or tear",
                                 "print or coating defect", "missing or broken part"],
                    "defects": ["dark particle suspended in the liquid",
                                "small insect trapped inside the vial",
                                "torn corner of the QR label",
                                "scratched-through print on the label",
                                "chip in the glass rim"],
                    "notes": 
"diffuse bright-field back light through a transparent vial of clear sparkling "
                             "liquid. Air bubbles, vial rotation, fill level and QR-code position all vary and are "
                             "normal, not defects. Real defects are small, low-contrast and often seen through "
                             "glass"},
    "wallplugs":   {"object": "a loose pile of plastic wall plugs on a plain background",
                    "families": ["missing or broken part", "crack or split", "deformation",
                                 "scratch or cut", "foreign body contamination"],
                    "defects": ["crack running down the shaft",
                                "snapped-off tip",
                                "missing flared collar",
                                "deformed misshapen plug",
                                "plastic burr left by the mould",
                                "deep scratch along the body"],
                    "notes": 
"diffuse bright-field front light on a loose pile of plastic wall plugs. They touch, "
                             "overlap and get cut off by the image frame, and their number and placement vary -- "
                             "all normal. Real defects sit on one individual plug"},
    "walnuts":     {"object": "several whole walnuts in their shells on a plain background",
                    "families": ["crack or split", "hole or tear", "missing or broken part",
                                 "mould or rot", "foreign body contamination"],
                    "defects": ["broken shell showing the kernel inside",
                                "crack splitting the shell open",
                                "hole bored through the shell exposing the kernel",
                                "patch of mould on the shell",
                                "dark speck of grit"],
                    "notes": 
"diffuse bright-field front light on loose walnuts. They touch, overlap and get cut "
                             "off by the image frame, and vary widely in size, shape and shell structure. The "
                             "natural shell grain is normal and is not a crack"},
}

# Optional: pin which families a category draws, in order, e.g.
#   FAMILY_PLAN = {"rice": ["foreign contamination", "stain or discolouration", "foreign object"]}
# Empty -> families are sampled per category using SEED.
FAMILY_PLAN = {}

# ---------------------------------------------------------------- prompt text (editable)
# Defect families: the diversity axis. Each image in a category is assigned a DIFFERENT
# family, so the images of one category cannot collapse onto the same defect.
#
# Grounded in MVTec AD 2 Table 3 ("Description of Occurring Defects"), which lists, across the
# eight categories: print defects, scratches, cuts, holes, colour inconsistencies, loose
# threads and extra fabric pieces, foreign object contamination of varying texture and size,
# (semi-)transparent plastic contamination, cracks, missing parts, broken pieces, and damaged
# or missing QR codes.
#
# The hints name CONCRETE industrial materials on purpose. Left open, a VLM proposes things
# like leaves or insects, which never occur on a sealed inspection rig -- what actually
# contaminates these lines is glass, plastic, metal swarf, grit, and fibres.
FAMILIES = {
    "scratch or cut":        "a thin abraded or incised line that breaks the surface finish "
                             "and exposes the material under it",
    "hole or tear":          "material missing right through, with visibly torn or cut edges",
    "crack or split":        "a fracture line in a rigid material, opening slightly along its length",
    "missing or broken part": "a piece of the item snapped off or absent, leaving a blunt "
                             "broken face",
    "foreign body contamination": "one small thing that does not belong -- a glass shard, a "
                             "plastic sliver, a metal shaving, a piece of grit, or on food "
                             "and liquid products a small insect",
    "fibre or thread contamination": "a stray thread or fibre lying across the surface",
    "print or coating defect": "printing or coating that is smeared, scratched through, "
                             "faded or misregistered",
    "stain or discolouration": "a localized patch where the colour has changed against the "
                             "material around it",
    "mould or rot":          "fungal growth or rotting on organic material -- a fuzzy or "
                             "darkened patch spreading across the surface",
    "deformation":           "warped, bent or misshapen geometry left by a moulding or "
                             "handling fault",
}

# Qualitative only. Numeric sizes are banned in the ask -- generators handle "tiny" more
# reliably than "3 mm", and a millimetre figure is meaningless without knowing the scale.
SIZE_WORDS = ["tiny", "very small", "small"]

# Frozen. Nothing here names a dataset or a category -- every category-specific string
# arrives from the VLM JSON.
# Two framings were tried and both failed in an instructive way.
#
#   "Generate the same image ... Keep everything else unchanged."
#       -> the preservation clause dominated and the generator returned the input untouched.
#
#   "Generate the same image in which X shows a defect"
#       -> "the same image" frames the job as copy-then-overlay, so the generator composited
#          a sprite ON TOP of the picture: a rock lying on the rice, an insect sitting on the
#          jelly. Nothing was damaged; something was pasted.
#
# The wording below asks for a natural anomaly IMAGE rather than an edit of an existing one,
# and then says outright that the defect belongs to the material. Preservation is still
# enforced -- by HARD_RULES 1 and 2, where it does not compete with the defect instruction.
TEMPLATE = (
    "Generate a natural-looking anomaly image of this scene: {target} has a {size} "
    "{defect}, {where}. "
    "The defect is part of the material, not laid on top -- its edges, depth and shadow "
    "follow the surface. "
    "Small, clearly visible, photorealistic."
)

HARD_RULES = """HARD RULES YOU MUST FOLLOW

1. Exact Image Preservation: Preserve the original image, geometry, composition, background, texture, lighting, shadows, reflections, perspective, and pixel correspondence everywhere except the requested defect.
2. Photometric & Dimensional Invariance: Preserve color, saturation, exposure, contrast, illumination, material appearance, image dimensions, aspect ratio, crop, framing, and resolution exactly; no resizing, cropping, padding, zooming, or global enhancement.
3. Single Localized Defect: Add exactly ONE small, category-valid, physically realistic anomaly; no reconstruction, recomposition, secondary defects, or any unrequested modification."""


# ---------------------------------------------------------------- VLM-1 wording (editable)
SYS_VLM1 = (
    "You are an industrial quality-inspection expert. Given a defect-free product image, "
    "you name one small realistic defect that could occur on it. You answer only with JSON."
)

ASK_VLM1 = """{dataset_context}

Look at this defect-free industrial inspection image.

  the object is: {object}

Describe ONE small, realistic defect of the following family that could appear on this exact
object during manufacturing or handling:

  family: {family} -- {family_hint}

Use what you can actually see: the material, the surface, the lighting, and how many items
are in frame. Refer to the object as described above, never as something else.
{constraints}{avoid}
Rules:
- If several items are in frame, "target" must say WHICH one, by position.
- "defect" is a short noun phrase of that family. It may add material, remove material, or
  alter something already present.
- It must be SMALL but UNMISTAKABLE -- a definite break in the surface with visible edges,
  not a faint tint or a subtle texture change. A scratch means the surface is torn open, not
  lightly shaded.
- NEVER propose something that is NORMAL for this product. Bubbles, reflections, specular
  highlights, shadows, natural grain or shell texture, print-pattern variation, and the
  ordinary arrangement or count of items are all normal and are NOT defects.
- "defect" must name a specific physical fault -- what happened to the material. "dark spot",
  "mark", "irregularity", "blemish", "anomaly", "imperfection" and "damage" are too vague and
  will be rejected. Say what it is: a torn thread, a bored hole, a shard of glass, a smear of
  ink.
- Phrase "defect" so it is PART OF the material, not resting on it. Say how it sits in the
  surface: "wedged between the grains", "pressed into the set jelly", "torn open across the
  weave", "gouged into the metal". Avoid wording that reads as an object simply placed on
  top.
- Anything foreign must be something that plausibly reaches this product on a production
  line: glass, plastic, metal, grit, thread or fibre, and on food or liquid products a small
  insect. Never leaves, twigs, or outdoor debris.
- Keep it local. Never whole-object damage, never a change to framing or lighting.
- Plain words. No sizes in numbers or millimetres, no metaphors, no optics terms like
  refract, specular, or diffuse.

The three fields are read into this sentence, so they must fit it grammatically:
  "... in which TARGET shows a {size} DEFECT, WHERE."

Reply with only JSON:
{{
  "defect_name": "<2-3 words, database key, e.g. 'shell crack', 'glass shard'>",
  "target":      "<what carries it, with article, e.g. 'the walnut left of centre'>",
  "defect":      "<short noun phrase, e.g. 'crack splitting the shell open', 'torn strip of label'>",
  "where":       "<short position phrase, e.g. 'near its upper seam'>"
}}"""


# ---------------------------------------------------------------- reply validation (editable)
# Plain word lists rather than regexes, so they can be edited without knowing regex syntax.
# They are compiled in the VLM-1 cell.
FIELDS = ("defect_name", "target", "defect", "where")

# Words that describe a SHAPE or a JUDGEMENT rather than a physical fault. A generator given
# "a small dark spot" has nothing to render; given "a bored hole" it does. Always rejected.
VAGUE_WORDS = ["spot", "mark", "blemish", "anomaly", "imperfection", "irregularity",
               "defect", "damage", "flaw", "issue", "artifact", "artefact"]

# Vague in general, legitimate for the stain family only.
STAIN_ONLY_WORDS = ["discolouration", "discoloration", "patch", "area", "region"]
STAIN_FAMILY     = "stain or discolouration"

# NORMAL variation in this data, never a defect. Bubbles in a vial, specular bands on
# dark-field metal and walnut shell grain are the trap MVTec AD 2 is built around.
NORMAL_WORDS = ["bubble", "bubbles", "reflection", "reflections", "highlight", "highlights",
                "glare", "shadow", "shadows", "grain pattern", "natural grain",
                "texture variation", "pattern variation", "lighting"]

# Ignored when checking that "target" names the object -- generic scene words, not nouns.
STOP_WORDS = ["with", "their", "plain", "background", "inside", "several", "single", "loose",
              "flat", "sheet", "pile", "filling", "frame", "lying", "and", "the", "its",
              "suspended", "machined", "whole", "shells", "strips", "bulk", "grains"]

# ---------------------------------------------------------------- generation (editable)
MAX_NEW_TOKENS = 200    # the JSON reply is short; this is headroom
TRIES          = 4      # attempts per image before the fallback fires
TEMP_GREEDY    = 0.0    # first attempt: deterministic
TEMP_SAMPLE    = 0.8    # retries and later images: sampled, so they diverge

CATS = list(CONFIG)
print(f"{len(CATS)} categories configured | source={DEFECT_SOURCE} | "
      f"{N_IMAGES} images per category -> {len(CATS) * N_IMAGES} prompts")
