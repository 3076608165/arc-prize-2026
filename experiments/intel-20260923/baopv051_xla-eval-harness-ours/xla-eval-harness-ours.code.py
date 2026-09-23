import logging

logger = logging.getLogger("eval_harness")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers.clear()
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_console)
logger.info("logger ready")

# %% ---

# ============================ SUA DUY NHAT DONG NAY ============================
MODEL_ID = "ours"   # "ours" | "bfree_official" | "globalforge_dinov2" | "globalforge_dinov3"
# ==============================================================================

import time

TIME_BUDGET_HOURS = 10.5          # ha xuong neu chay song song >=2 notebook cung luc
_T0 = time.time()

def time_left_hours():
    return TIME_BUDGET_HOURS - (time.time() - _T0) / 3600

def budget_ok(margin_hours=0.3):
    ok = time_left_hours() > margin_hours
    if not ok:
        logger.warning(f"[BUDGET] con {time_left_hours():.2f}h < margin {margin_hours}h -> DUNG phan con lai")
    return ok

# Benchmarks. synthbuster_original / genimage_val DA BI LOAI (trung voi ai_genbench).
BENCHMARKS_ENABLED = {
    "ai_genbench":         True,   # thay the synthbuster_original + genimage_val
    "wildrf":              True,   # chi chay neu WILDRF_DIR duoc dien o cell PATHS
    "synthwildx":          True,   # chi chay neu SYNTHWILDX_DIR duoc dien
    "bfree_viral_images":  True,   # chi chay neu BFREE_VIRAL_DIR duoc dien
    "realdeg_bench_style": False,  # 13 dieu kien degradation, chay tren cac bo o duoi
}

IMG_SIZE = 504
DEVICE = "cuda:0"
BATCH_SIZE = 8
logger.info(f"MODEL_ID={MODEL_ID} | budget={TIME_BUDGET_HOURS}h | benchmarks={BENCHMARKS_ENABLED}")

# %% ---

from pathlib import Path

WORK = Path("/kaggle/working")

# ---------- repo / code ----------
BFREE_SRC_DIR        = Path("/kaggle/input/notebooks/baopv051/xla-setup/bfree_src")            # repo CUA BAN
BFREE_SRC_CODE       = BFREE_SRC_DIR / "code"                                                   # chua networks/, configs/, train_lora.py
OFFLINE_BUNDLE       = Path("/kaggle/input/notebooks/baopv051/xla-setup-eval/offline_bundle")
BFREE_OFFICIAL_REPO  = OFFLINE_BUNDLE / "repos" / "bfree_official"
BFREE_OFFICIAL_CODE  = BFREE_OFFICIAL_REPO / "code"
GLOBALFORGE_REPO     = OFFLINE_BUNDLE / "repos" / "globalforge_official"
GLOBALFORGE_CODE     = GLOBALFORGE_REPO / "code"
GLOBALFORGE_CKPT_DIR = GLOBALFORGE_REPO / "ckpt"                                                # checkpoint-best.pth (hoac *.part_*)
GLOBALFORGE_WEIGHTS_ROOT = OFFLINE_BUNDLE / "weights"                                           # chua <mode>/config.json (backbone HF)

# ---------- wheels (cai offline) ----------
WHEELS_DIR       = Path("/kaggle/input/notebooks/baopv051/xla-setup/wheels_rtxpro6000")
WHEELS_EXTRA_DIR = OFFLINE_BUNDLE / "wheels_eval_extra"

# ---------- weights ----------
OURS_CKPT              = Path("/kaggle/input/notebooks/baopv051/xla-training-phase-2/bfree_globalforge_lora_r16_best.pth")
BFREE_OFFICIAL_WEIGHTS = Path("/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1")

# ---------- datasets ----------
AI_GENBENCH_DIR   = Path("/kaggle/input/datasets/baopv051/ai-genbench-v1")   # HF Arrow (dataset_dict.json + validation/*.arrow)
WILDRF_DIR        = Path("/kaggle/input/datasets/ngovanphuong2604/wildrf-dataset")         # -> .../{test,train,val}/<platform>/{0_real,1_fake}
SYNTHWILDX_DIR    = Path("/kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset")     # -> HF Arrow
BFREE_VIRAL_DIR   = Path("/kaggle/input/datasets/ngovanphuong2604/bfree-viral-images")      # -> .../{FAKE,REAL}/<album_id>/*  (FIX: truoc do bi trung voi SYNTHWILDX_DIR)

# ---------- AI-GenBench: split / sampling / chong trung ----------
AIGENBENCH_SPLIT            = "validation"
AIGENBENCH_MAX_PER_GEN      = 150    # so anh FAKE toi da moi generator (None = lay het)
AIGENBENCH_MIN_PER_GEN      = 20     # generator it hon nguong nay -> bo qua (metric vo nghia)
AIGENBENCH_MAX_REAL         = 1500   # pool REAL dung chung cho moi generator (score 1 lan)
AIGENBENCH_SEED             = 42
# Loai origin_dataset TRUNG voi tap TRAIN cua ban (khop theo substring, khong phan biet hoa thuong).
# Vi du neu ban train tren DRCT + COCO: AIGENBENCH_EXCLUDE_ORIGIN = ["DRCT", "COCO"]
AIGENBENCH_EXCLUDE_ORIGIN   = []
# Loai luon generator theo ten neu can (vd da dung chinh no de train).
AIGENBENCH_EXCLUDE_GENERATOR = []

WILDRF_SPLIT = "test"


def require_dir(p, what):
    p = Path(p)
    if not p.is_dir():
        raise FileNotFoundError(f"NOT FOUND (thu muc): {what}\n  mong doi: {p}\n"
                                "  -> Attach dung input, hoac sua duong dan o cell CONFIG-PATHS.")
    return p


def require_file(p, what):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f"NOT FOUND (file): {what}\n  mong doi: {p}\n"
                                "  -> Attach dung input, hoac sua duong dan o cell CONFIG-PATHS.")
    return p


def optional_dir(p):
    return Path(p) if (p is not None and Path(p).is_dir()) else None


for label, p in [("bfree_src/code (repo cua ban)", BFREE_SRC_CODE),
                 ("offline_bundle", OFFLINE_BUNDLE),
                 ("wheels_rtxpro6000", WHEELS_DIR),
                 ("ai-genbench", AI_GENBENCH_DIR)]:
    logger.info(f"[path] {label}: {p} -> {'OK' if Path(p).is_dir() else 'THIEU'}")

# %% ---

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
logger.info("offline env vars set")

# %% ---

import subprocess
import sys


_find_links = ["--find-links", str(WHEELS_DIR)]
if WHEELS_EXTRA_DIR.is_dir():
    _find_links += ["--find-links", str(WHEELS_EXTRA_DIR)]
else:
    logger.warning(f"wheels_eval_extra khong ton tai ({WHEELS_EXTRA_DIR}) - chi cai tu wheels goc")


def _wheel_available(pkg_name, version=None):
    """Tim file .whl khop ten (+ version neu co) trong WHEELS_DIR / WHEELS_EXTRA_DIR,
    khong dung index nao khac. Tra ve True/False - dung de QUYET DINH truoc khi goi
    pip, tranh de pip tu bao loi 'Could not find a version' roi dung ca lenh."""
    norm = pkg_name.lower().replace("-", "_")
    for d in (WHEELS_DIR, WHEELS_EXTRA_DIR):
        if not d.is_dir():
            continue
        for whl in d.glob("*.whl"):
            name = whl.name.lower()
            if not name.startswith(norm + "-") and not name.startswith(norm.replace("_", "-") + "-"):
                continue
            if version is None or f"-{version}-" in name or name.startswith(f"{norm}-{version}"):
                return True
    return False


def pip_install_latest(pkg):
    """Cai ban moi nhat co trong wheels dir (khong pin version); neu khong co wheel
    nao, GIU ban co san cua he thong (khong goi pip, tranh loi resolver)."""
    if not _wheel_available(pkg):
        logger.info(f"{pkg}: khong co wheel trong wheels dir -> dung ban co san cua he thong.")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    *_find_links, pkg], check=False)


# ---- Bundle huggingface_hub/transformers/tokenizers/accelerate/peft/timm ----
_HF_BUNDLE = {
    "timm": "1.0.22",
    "peft": "0.15.2",
    "huggingface_hub": "0.36.0",
    "tokenizers": "0.21.1",
    "accelerate": "1.3.0",
    "transformers": "4.56.2",
}
_missing = [f"{p}=={v}" for p, v in _HF_BUNDLE.items() if not _wheel_available(p, v)]
if _missing:
    logger.warning(
        "Bundle huggingface_hub/transformers/... KHONG day du trong wheels dir "
        f"(thieu: {_missing}) -> BO QUA CA BUNDLE, giu nguyen stack co san cua he "
        "thong de tranh mismatch API (vd is_offline_mode). Neu globalforge_dinov3 can "
        "Dinov3Model (transformers>=4.56) ma he thong khong co, load se bao loi ro rang.")
else:
    logger.info("Bundle huggingface_hub/transformers/... du wheel -> cai dong bo ca bo.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_find_links,
                    *[f"{p}=={v}" for p, v in _HF_BUNDLE.items()]], check=False)

pip_install_latest("safetensors")
pip_install_latest("datasets")

import json
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

logger.info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

import transformers as _tf

# try:
#     import peft.tuners.lora.torchao as _peft_lora_torchao
#     _peft_lora_torchao.dispatch_torchao = lambda *a, **kw: None
#     logger.info("da vo hieu hoa peft dispatch_torchao (torchao cu/khong can thiet cho LoRA nn.Linear)")
# except ImportError:
#     pass 

try:
    import datasets as _hf_datasets
    logger.info(f"datasets={_hf_datasets.__version__} (BAT BUOC cho ai_genbench / synthwildx)")
except ImportError:
    logger.error("Package 'datasets' KHONG co san -> ai_genbench va synthwildx se bi SKIP. "
                 "Them wheel 'datasets' (+ pyarrow/dill/xxhash/multiprocess) vao wheels_eval_extra.")

# %% ---

class ModelWrapper:
    """Giao dien thong nhat: .score(list[PIL.Image]) -> np.ndarray[float] xac suat fake."""

    def __init__(self, model, normalize, img_size, forward_fn, multi_crop=False):
        self.model = model.to(DEVICE).eval()
        self.normalize = normalize
        self.img_size = img_size
        self.forward_fn = forward_fn
        self.multi_crop = multi_crop

    @torch.no_grad()
    def score(self, pil_images):
        probs = []
        for img in pil_images:
            img = img.convert("RGB")
            w, h = img.size
            if min(w, h) < self.img_size:
                scale = self.img_size / min(w, h)
                img = img.resize((max(self.img_size, round(w * scale)),
                                  max(self.img_size, round(h * scale))), Image.BICUBIC)
            crops = self._crops(img)
            batch = torch.stack([self.normalize(c) for c in crops]).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.forward_fn(self.model, batch)
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].mean().item())
        return np.array(probs)

    def _crops(self, img):
        w, h = img.size
        s = self.img_size
        if not self.multi_crop or (w == s and h == s):
            left, top = (w - s) // 2, (h - s) // 2
            return [img.crop((left, top, left + s, top + s))]
        xs = {max(0, x) for x in (0, w - s, (w - s) // 2)}
        ys = {max(0, y) for y in (0, h - s, (h - s) // 2)}
        return [img.crop((x, y, x + s, y + s)) for x in xs for y in ys]


def _resnet_normalize():
    return T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _raw_normalize():
    # GlobalForge REM tu normalize BEN TRONG forward -> chi ToTensor, tranh double-norm.
    return T.Compose([T.ToTensor()])


_BFREE_NORM_MAP = {
    "resnet":   T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    "clip":     T.Compose([T.ToTensor(), T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                     std=(0.26862954, 0.26130258, 0.27577711))]),
    "none":     T.Compose([T.ToTensor()]),
    "xception": T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
}


# =============================== ours ===============================
def _load_ours():
    require_dir(BFREE_SRC_CODE, "bfree_src/code (repo cua BAN)")
    require_dir(BFREE_SRC_CODE / "networks", "bfree_src/code/networks")
    require_file(OURS_CKPT, "checkpoint 'ours' (output notebook training)")

    # KHONG dung train_lora.py (apply_lora_to_backbone -> peft get_peft_model).
    # Tren Kaggle, peft moi (dispatch_torchao) bat buoc torchao > 0.16.0 nhung
    # env co 0.10.0 -> ImportError khi boc LoRA. Vi cung chi can LOAD + MERGE
    # (tuong duong merge_and_unload()), nen merge LoRA thu cong bang PyTorch
    # thuan: W = W + (alpha/r) * B @ A. networks.bfree_globalforge_vit chi
    # import torch/timm/modules - khong import peft/train_lora/datasets ->
    # khong can hack cach ly cache 'datasets*' nua (hack do chi ton tai vi
    # train_lora.py import datasets.bfree_dataset trung ten HF "datasets").
    sys.path.insert(0, str(BFREE_SRC_CODE))
    try:
        from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
        from configs.loader import build_model_kwargs
    finally:
        sys.path.remove(str(BFREE_SRC_CODE))

    ckpt = torch.load(OURS_CKPT, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and not isinstance(ckpt["model"], torch.Tensor):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # FIX KeyError 'backbone': CONFIG trong ckpt la dict PHANG cua notebook train
    # (02_bfree_kaggle_train.ipynb:683) - khong phai cau truc long cua
    # bfree_dcs.yaml (backbone:{...}, loss:{...}) ma build_model_kwargs() doi hoi.
    # State_dict ckpt co lib.*/gsr.* -> use_lib=True, use_gsr=True (da xac minh).
    def _flat_cfg_to_nested(flat):
        return {
            "backbone": {
                "arch": flat["arch"],
                "num_classes": flat.get("num_classes", 2),
                "img_size": flat.get("img_size", 504),
                "pretrained": False,
                "use_lib": flat.get("use_lib", True),
                "use_gsr": flat.get("use_gsr", True),
                "lib_kernel": flat.get("lib_kernel", 3),
                "lib_tau": flat.get("lib_tau", 0.5),
                "gsr_window": flat.get("gsr_window", 3),
                "gsr_mask_prob": flat.get("gsr_mask_prob", 1.0),
            },
            "loss": {
                "use_dcs": flat.get("use_dcs", True),
                "dcs_tau": flat.get("dcs_tau", 0.07),
                "lambda_dcs": flat.get("lambda_dcs", 0.1),
                "label_smoothing": flat.get("label_smoothing", 0.1),
            },
        }

    def _merge_stem(stem):
        prefix_old = "model.base_model.model."
        stem = ("model." + stem[len(prefix_old):]) if stem.startswith(prefix_old) else stem
        return stem.replace(".base_layer.", ".")

    default_cfg = {
        "backbone": {"arch": "vit_base_patch14_reg4_dinov2.lvd142m", "num_classes": 2,
                     "img_size": 504, "pretrained": False, "use_lib": True, "use_gsr": True,
                     "lib_kernel": 3, "lib_tau": 0.5, "gsr_window": 3, "gsr_mask_prob": 1.0},
        "loss": {"use_dcs": True, "lambda_dcs": 0.1, "dcs_tau": 0.07, "label_smoothing": 0.1},
    }
    cfg = ckpt.get("config", default_cfg) if isinstance(ckpt, dict) else default_cfg
    flat_cfg = cfg if isinstance(cfg, dict) and "backbone" not in cfg else None
    if flat_cfg is not None:
        cfg = _flat_cfg_to_nested(flat_cfg)
    kwargs = build_model_kwargs(cfg)
    kwargs["pretrained"] = False  # load tu ckpt - tranh timm goi HF download khi offline
    model = BFreeGlobalForgeViT(**kwargs)

    # Merge LoRA thu cong (khong PEFT). State_dict ckpt co prefix
    # 'model.base_model.model.*' (PEFT wrapper) + 'base_layer.*' (module goc bi
    # doi ten) + 49 cap 'lora_A/lora_B.default.weight' (Da XAC MINH voi ckpt
    # that: Linear A=(r,in), B=(out,r); Conv2d patch A=(r,3,14,14), B=(out,r,1,1);
    # scaling = lora_alpha/r = 32/16 = 2.0). Cong thuc delta =
    # (B.view(out,-1) @ A.view(r,-1)).view(W.shape) dung chung cho Linear va
    # Conv2d k1x1. Sau merge, model la VisionTransformer thuan - khong can
    # merge_and_unload().
    lora_alpha = flat_cfg["lora_alpha"] if flat_cfg is not None else 32
    lora_rank = flat_cfg.get("lora_rank", 16) if flat_cfg is not None else 16
    scaling = lora_alpha / lora_rank

    lora_pairs = {}
    for k, v in sd.items():
        if k.endswith(".lora_A.default.weight"):
            lora_pairs.setdefault(_merge_stem(k[:-len(".lora_A.default.weight")]) + ".weight", {})["A"] = v
        elif k.endswith(".lora_B.default.weight"):
            lora_pairs.setdefault(_merge_stem(k[:-len(".lora_B.default.weight")]) + ".weight", {})["B"] = v

    merged = {}
    n_merged = 0
    for k, v in sd.items():
        if ".lora_A.default." in k or ".lora_B.default." in k:
            continue
        k2 = _merge_stem(k)
        if k2 in lora_pairs:
            A, B = lora_pairs[k2]["A"], lora_pairs[k2]["B"]
            merged[k2] = v + scaling * (B.view(B.shape[0], -1) @ A.view(A.shape[0], -1)).view(v.shape)
            n_merged += 1
        else:
            merged[k2] = v
    if n_merged:
        logger.info(f"ours: merged {n_merged} LoRA layers (scaling={scaling})")

    model.load_state_dict(merged, strict=True)

    def forward_fn(m, x):
        # BFreeGlobalForgeViT.forward tra dict {"logits","cls"} - khong co forward_eval;
        # khong dung compute_loss vi ham do doi clean+degraded pair + label.
        return m(x)["logits"]

    return ModelWrapper(model, _resnet_normalize(), IMG_SIZE, forward_fn, multi_crop=True)

# =========================== bfree_official ===========================
BFREE_OFFICIAL_FALLBACK_ARCH = "timm_c5i504_vit_base_patch14_reg4_dinov2.lvd142m"
BFREE_OFFICIAL_FALLBACK_NORM = "clip"


def _load_bfree_official():
    require_dir(BFREE_OFFICIAL_CODE, "bfree_official/code (offline_bundle)")
    require_dir(BFREE_OFFICIAL_WEIGHTS, "weights goc B-Free")
    sys.path.insert(0, str(BFREE_OFFICIAL_CODE))
    from networks import get_network, load_weights
    import yaml

    # Chi tim BEN TRONG BFREE_OFFICIAL_WEIGHTS (1 root duy nhat), khong quet /kaggle/input.
    cfg_path, data = None, None
    for hit in sorted(BFREE_OFFICIAL_WEIGHTS.rglob("config.yaml")):
        try:
            d = yaml.safe_load(hit.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "weights_file" in d and (hit.parent / d["weights_file"]).is_file():
            cfg_path, data = hit, d
            break

    if cfg_path is not None:
        weights_path = cfg_path.parent / data["weights_file"]
        arch = data.get("arch", BFREE_OFFICIAL_FALLBACK_ARCH)
        norm = data.get("norm_type", BFREE_OFFICIAL_FALLBACK_NORM)
    else:
        cands = sorted(list(BFREE_OFFICIAL_WEIGHTS.rglob("*.pth")) +
                       list(BFREE_OFFICIAL_WEIGHTS.rglob("*.pt")) +
                       list(BFREE_OFFICIAL_WEIGHTS.rglob("*.tar")))
        if not cands:
            raise FileNotFoundError(
                f"NOT FOUND: khong thay config.yaml lan checkpoint nao duoi {BFREE_OFFICIAL_WEIGHTS}")
        weights_path = cands[0]
        arch, norm = BFREE_OFFICIAL_FALLBACK_ARCH, BFREE_OFFICIAL_FALLBACK_NORM
        logger.warning(f"bfree_official: KHONG co config.yaml -> dung fallback arch={arch} norm={norm}. "
                       "Kiem tra lai neu load_state_dict bao mismatch.")

    logger.info(f"bfree_official: {weights_path} | arch={arch} | norm={norm}")
    model = load_weights(get_network(arch), str(weights_path)).to(DEVICE).eval()

    def forward_fn(m, x):
        l = m(x)                      # num_classes=1 -> pseudo 2-logit: softmax[:,1] = sigmoid(l)
        return torch.cat([-l, l], dim=1)

    # Wrapper5crops tu lam 5-crop ben trong forward -> multi_crop=False de tranh crop kep.
    return ModelWrapper(model, _BFREE_NORM_MAP.get(norm, _BFREE_NORM_MAP["clip"]),
                        IMG_SIZE, forward_fn, multi_crop=False)


# ============================ globalforge ============================
def _resolve_globalforge_ckpt():
    require_dir(GLOBALFORGE_CKPT_DIR, "globalforge_official/ckpt")
    direct = GLOBALFORGE_CKPT_DIR / "checkpoint-best.pth"
    if direct.is_file():
        return direct
    parts = sorted(GLOBALFORGE_CKPT_DIR.glob("*.part_*"))
    if parts:
        merged = WORK / "checkpoint-best.pth"
        if not merged.is_file():
            import shutil
            with open(merged, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out, 1024 * 1024)
            logger.info(f"globalforge: ghep {len(parts)} part -> {merged}")
        return merged
    cands = sorted(GLOBALFORGE_CKPT_DIR.glob("*.pth"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"NOT FOUND: checkpoint GlobalForge duoi {GLOBALFORGE_CKPT_DIR}")


def _load_globalforge(variant):  # "dinov2" | "dinov3"
    require_dir(GLOBALFORGE_CODE, "globalforge_official/code")
    sys.path.insert(0, str(GLOBALFORGE_CODE))
    mode = {"dinov2": "vit_l_b", "dinov3": "vit_l_a"}[variant]

    if variant == "dinov3":
        try:
            import transformers as _tf
            _ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
            if _ver < (4, 56):
                logger.warning(f"globalforge_dinov3: transformers={_tf.__version__} < 4.56 - "
                               "ban co the KHONG co Dinov3Model trong AutoModel. Neu import "
                               "REM/backbone that bai o duoi, day la nguyen nhan; can wheel "
                               "transformers>=4.56 trong wheels_eval_extra.")
        except ImportError:
            pass

    require_dir(GLOBALFORGE_WEIGHTS_ROOT, "offline_bundle/weights (backbone HF-format)")
    require_file(GLOBALFORGE_WEIGHTS_ROOT / mode / "config.json",
                 f"backbone HF-format GlobalForge mode={mode}")
    # REM doc env GLOBALFORGE_WEIGHTS_DIR luc IMPORT -> set TRUOC khi import models.REM.
    os.environ["GLOBALFORGE_WEIGHTS_DIR"] = str(GLOBALFORGE_WEIGHTS_ROOT)
    logger.info(f"globalforge: weights_root={GLOBALFORGE_WEIGHTS_ROOT} | mode={mode}")

    ckpt_path = _resolve_globalforge_ckpt()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    use_lib = any(k.startswith("lib.") for k in state)
    use_gsr = any(k.startswith("gsr.") for k in state)
    lora_rank = 16
    for k, v in state.items():
        if "lora_A.default.weight" in k:
            lora_rank = int(v.shape[0])
            break

    from models.REM import REM
    if lora_rank != 16:
        def _patched_lora(self):
            from peft import LoraConfig, get_peft_model as _gpm
            qkv = ["q_proj", "k_proj", "v_proj"] if self.vision == 3 else ["query", "key", "value"]
            config = LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, target_modules=qkv)
            for i in range(0, 24):
                layer = self.backbone.layer[i] if self.vision == 3 else self.backbone.encoder.layer[i]
                self.backbone.layer[i] = _gpm(layer, config)
        REM.HFViTBackbone._lora = _patched_lora

    model = REM(mode=mode, use_lib=use_lib, use_gsr=use_gsr,
                lib_layer=0, gsr_layer=0, lib_kernel=3, lib_tau=0.5,
                gsr_window=3, gsr_mask_prob=1.0)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    logger.info(f"globalforge: ckpt={ckpt_path.name} use_lib={use_lib} use_gsr={use_gsr} lora_rank={lora_rank}")

    return ModelWrapper(model, _raw_normalize(), IMG_SIZE,
                        lambda m, x: m(x)["logits"], multi_crop=False)


def load_model(model_id):
    if model_id == "ours":
        return _load_ours()
    if model_id == "bfree_official":
        return _load_bfree_official()
    if model_id == "globalforge_dinov2":
        return _load_globalforge("dinov2")
    if model_id == "globalforge_dinov3":
        return _load_globalforge("dinov3")
    raise ValueError(model_id)


MODEL = load_model(MODEL_ID)
logger.info(f"model '{MODEL_ID}' loaded OK")

# %% ---

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve

def binary_ece(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def balanced_nll(y_true, y_prob, eps=1e-7):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    nll_pos = -np.log(y_prob[y_true == 1]).mean() if (y_true == 1).any() else np.nan
    nll_neg = -np.log(1 - y_prob[y_true == 0]).mean() if (y_true == 0).any() else np.nan
    return float(np.nanmean([nll_pos, nll_neg]))


def find_optimal_threshold(y_true, y_prob, method="youden"):
    """Nguong toi uu tren xac suat p_fake in [0,1] (KHONG phai logit).
    Youden's J = argmax(TPR - FPR) tren duong ROC — chuan cho bai toan balanced accuracy
    vi Youden's J toi da hoa dung TPR+TNR-1, tuong duong toi da hoa balanced accuracy."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if method == "youden":
        j = tpr - fpr
        best = int(np.argmax(j))
        # roc_curve co the tra thr[0] = inf (diem dau) - kep ve [0,1] cho an toan
        return float(np.clip(thr[best], 0.0, 1.0))
    raise ValueError(method)


# Nguong hieu chinh toan cuc — mac dinh = 0.5 (giong het truoc day) cho toi khi
# run_ai_genbench() tinh lai tu pool ai_genbench (xem cell chay chinh ben duoi).
# De MOT bien global duy nhat de moi loi goi compute_all_metrics() sau do (wildrf,
# bfree_viral_images, synthwildx, realdeg...) TU DONG dung chung nguong nay, khong
# can sua tung ham goi rieng le.
CALIB_THRESHOLD = 0.5
CALIB_SOURCE = "default (chua hieu chinh)"


def compute_all_metrics(y_true, y_prob, threshold=None):
    """threshold=None -> dung CALIB_THRESHOLD hien tai (global, cap nhat boi run_ai_genbench).
    Luon tra CA HAI: 'bacc' (nguong co dinh 0.5, giu de so sanh voi cac lan chay truoc) va
    'bacc_calib' (nguong hieu chinh) + 'thr_calib' (gia tri nguong da dung)."""
    y_true = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    thr_calib = CALIB_THRESHOLD if threshold is None else threshold

    y_pred_05 = (y_prob_arr >= 0.5).astype(int)
    y_pred_calib = (y_prob_arr >= thr_calib).astype(int)

    out = {"n": int(len(y_true)), "n_real": int((y_true == 0).sum()), "n_fake": int((y_true == 1).sum())}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob_arr))
    except ValueError:
        out["auc"] = float("nan")
    out["bacc"] = float(balanced_accuracy_score(y_true, y_pred_05)) * 100
    out["bacc_calib"] = float(balanced_accuracy_score(y_true, y_pred_calib)) * 100
    out["thr_calib"] = float(thr_calib)
    out["ece"] = binary_ece(y_true, y_prob_arr)
    out["nll"] = balanced_nll(y_true, y_prob_arr)
    return out


# %% ---

import io
import random
from PIL import ImageFilter, ImageEnhance

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".JPEG"}


def list_images(d):
    d = Path(d)
    if not d.is_dir():
        return []
    return [p for p in d.rglob("*") if p.suffix.lower() in {e.lower() for e in IMG_EXTS}]


def _find_child_with(root, required_subdirs):
    """BFS tu root tim thu muc dau tien co DU cac thu muc con yeu cau (bo qua wrapper long nhau)."""
    from collections import deque
    queue, seen = deque([Path(root)]), set()
    while queue:
        d = queue.popleft()
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        if all((d / s).is_dir() for s in required_subdirs):
            return d
        try:
            queue.extend(p for p in d.iterdir() if p.is_dir())
        except OSError:
            continue
    return None


class ArrowSource:
    """Boc 1 datasets.Dataset: .get(i) -> PIL.Image (decode lazy)."""

    def __init__(self, ds, img_col):
        self.ds = ds
        self.img_col = img_col

    def __len__(self):
        return len(self.ds)

    def get(self, idx):
        return self.ds[int(idx)][self.img_col]


def load_pil(source, key):
    img = Image.open(key) if source is None else source.get(key)
    return img.convert("RGB")


def score_items(items, source=None, batch_size=None, degrade_op=None, degrade_strength=None,
                compound_n=None, rng=None, tag=""):
    """items: list[(key, label)] -> (y_true, y_prob). Bo qua anh loi doc, co log."""
    batch_size = batch_size or BATCH_SIZE
    y_true, y_prob, n_err = [], [], 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs, labels = [], []
        for key, lab in chunk:
            try:
                img = load_pil(source, key)
            except Exception as e:
                n_err += 1
                if n_err <= 3:
                    logger.warning(f"[{tag}] bo qua anh loi ({key}): {e}")
                continue
            if compound_n is not None:
                img = apply_compound(img, compound_n, rng)
            elif degrade_op is not None:
                img = apply_degradation(img, degrade_op, degrade_strength)
            imgs.append(img)
            labels.append(lab)
        if not imgs:
            continue
        y_prob.extend(MODEL.score(imgs).tolist())
        y_true.extend(labels)
    if n_err:
        logger.warning(f"[{tag}] tong cong {n_err} anh bi bo qua do loi doc")
    return y_true, y_prob


def eval_items(items, source=None, **kw):
    return compute_all_metrics(*score_items(items, source=source, **kw))

# %% ---

DEGRADE_STRENGTHS = {
    "jpeg":       [90, 80, 70, 60, 40],
    "blur":       [0.5, 1, 2, 3, 5],
    "resize":     [0.9, 0.7, 0.5, 0.3, 0.2],
    "noise":      [0.0005, 0.001, 0.002, 0.005, 0.01],
    "brightness": [-0.2, -0.1, 0.1, 0.2],
    "contrast":   [-0.3, -0.2, 0.1, 0.2],
    "saturation": [0.6, 0.8, 1.3, 1.5],
}
DEGRADE_OPS = list(DEGRADE_STRENGTHS.keys())


def apply_degradation(img, op, strength):
    if op == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(strength))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=strength))
    if op == "resize":
        w, h = img.size
        small = img.resize((max(1, int(w * strength)), max(1, int(h * strength))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if op == "noise":
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0, np.sqrt(strength), arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if op == "brightness":
        return ImageEnhance.Brightness(img).enhance(1 + strength)
    if op == "contrast":
        return ImageEnhance.Contrast(img).enhance(1 + strength)
    if op == "saturation":
        return ImageEnhance.Color(img).enhance(strength)
    raise ValueError(op)


def apply_compound(img, n_ops, rng):
    for _ in range(n_ops):
        op = rng.choice(DEGRADE_OPS)
        img = apply_degradation(img, op, rng.choice(DEGRADE_STRENGTHS[op]))
    return img


def realdeg_conditions():
    conds = [("clean", None)]
    conds += [(f"single_{op}", op) for op in DEGRADE_OPS]
    conds += [(f"compound_step{n}", n) for n in range(1, 6)]
    return conds

# %% ---

AIGB_CACHE = {}


def _aigenbench_dataset():
    if "ds" in AIGB_CACHE:
        return AIGB_CACHE["ds"]
    require_dir(AI_GENBENCH_DIR, "ai-genbench (HF Arrow)")
    from datasets import load_from_disk

    obj = load_from_disk(str(AI_GENBENCH_DIR))
    if hasattr(obj, "keys"):          # DatasetDict
        splits = list(obj.keys())
        split = AIGENBENCH_SPLIT if AIGENBENCH_SPLIT in splits else splits[0]
        if split != AIGENBENCH_SPLIT:
            logger.warning(f"ai_genbench: khong co split '{AIGENBENCH_SPLIT}', dung '{split}' (co: {splits})")
        ds = obj[split]
    else:
        split, ds = "(single)", obj

    img_col = next((c for c, f in ds.features.items() if f.__class__.__name__ == "Image"), None)
    if img_col is None:
        raise RuntimeError(f"ai_genbench: khong tim thay cot anh. features={list(ds.features)}")

    label_feat = ds.features.get("label")
    label_names = list(getattr(label_feat, "names", []) or [])
    logger.info(f"ai_genbench: split={split} n={len(ds)} img_col='{img_col}' label_names={label_names}")

    AIGB_CACHE["ds"] = (ds, img_col, label_names)
    return AIGB_CACHE["ds"]


def _aigenbench_meta(ds, label_names):
    """DataFrame metadata (khong decode anh) + cot 'y' nhi phan (1=fake)."""
    if "meta" in AIGB_CACHE:
        return AIGB_CACHE["meta"]
    cols = [c for c in ("label", "generator", "origin_dataset") if c in ds.column_names]
    try:
        meta = ds.select_columns(cols).to_pandas()
    except AttributeError:   # datasets cu
        meta = ds.remove_columns([c for c in ds.column_names if c not in cols]).to_pandas()

    if label_names:
        fake_ids = {i for i, n in enumerate(label_names) if "fake" in str(n).lower()}
        if not fake_ids:
            fake_ids = {len(label_names) - 1}
        meta["y"] = meta["label"].apply(lambda v: 1 if int(v) in fake_ids else 0)
    else:
        meta["y"] = (meta["label"].astype(int) != 0).astype(int)

    if "generator" not in meta.columns:
        meta["generator"] = "unknown"
    if "origin_dataset" not in meta.columns:
        meta["origin_dataset"] = "unknown"
    meta["generator"] = meta["generator"].fillna("unknown").astype(str).str.strip()
    meta["origin_dataset"] = meta["origin_dataset"].fillna("unknown").astype(str)

    logger.info(f"ai_genbench: meta OK | real={int((meta.y == 0).sum())} fake={int((meta.y == 1).sum())} "
                f"| n_generator={meta.loc[meta.y == 1, 'generator'].nunique()}")
    AIGB_CACHE["meta"] = meta
    return meta


def build_ai_genbench_groups():
    """-> (source, real_items, {generator: fake_items}) da subsample + loai trung."""
    ds, img_col, label_names = _aigenbench_dataset()
    meta = _aigenbench_meta(ds, label_names)
    rng = np.random.default_rng(AIGENBENCH_SEED)

    keep = np.ones(len(meta), dtype=bool)
    if AIGENBENCH_EXCLUDE_ORIGIN:
        low = meta["origin_dataset"].str.lower()
        drop = np.zeros(len(meta), dtype=bool)
        for pat in AIGENBENCH_EXCLUDE_ORIGIN:
            drop |= low.str.contains(pat.lower(), regex=False).to_numpy()
        keep &= ~drop
        logger.info(f"ai_genbench: loai {int(drop.sum())} row trung origin_dataset {AIGENBENCH_EXCLUDE_ORIGIN}")
    if AIGENBENCH_EXCLUDE_GENERATOR:
        excl = {g.lower() for g in AIGENBENCH_EXCLUDE_GENERATOR}
        drop = meta["generator"].str.lower().isin(excl).to_numpy()
        keep &= ~drop
        logger.info(f"ai_genbench: loai {int(drop.sum())} row theo generator {AIGENBENCH_EXCLUDE_GENERATOR}")

    idx_all = np.arange(len(meta))
    y = meta["y"].to_numpy()

    real_idx = idx_all[keep & (y == 0)]
    if AIGENBENCH_MAX_REAL and len(real_idx) > AIGENBENCH_MAX_REAL:
        real_idx = rng.choice(real_idx, AIGENBENCH_MAX_REAL, replace=False)
    real_items = [(int(i), 0) for i in sorted(real_idx)]

    groups, skipped = {}, []
    gens = meta.loc[keep & (y == 1), "generator"]
    for gen, sub in gens.groupby(gens):           # gop moi generator lam 1 nhom (du khac origin_dataset)
        gi = sub.index.to_numpy()
        if len(gi) < AIGENBENCH_MIN_PER_GEN:
            skipped.append((gen, len(gi)))
            continue
        if AIGENBENCH_MAX_PER_GEN and len(gi) > AIGENBENCH_MAX_PER_GEN:
            gi = rng.choice(gi, AIGENBENCH_MAX_PER_GEN, replace=False)
        groups[str(gen)] = [(int(i), 1) for i in sorted(gi)]

    if skipped:
        logger.info(f"ai_genbench: bo qua {len(skipped)} generator < {AIGENBENCH_MIN_PER_GEN} anh: "
                    f"{[g for g, _ in skipped][:10]}{' ...' if len(skipped) > 10 else ''}")
    logger.info(f"ai_genbench: {len(groups)} generator | real_pool={len(real_items)} | "
                f"tong fake={sum(len(v) for v in groups.values())}")
    return ArrowSource(ds, img_col), real_items, groups

# %% ---

def build_benchmark_filelist(name):
    """-> dict {group_name: [(path, label), ...]} cho cac bo dang ANH ROI."""
    if name == "wildrf":
        root = optional_dir(WILDRF_DIR)
        if root is None:
            raise FileNotFoundError("wildrf: chua dien WILDRF_DIR o cell CONFIG-PATHS")
        split_dir = None
        for hit in root.rglob(WILDRF_SPLIT):
            if hit.is_dir() and any((p / "0_real").is_dir() or (p / "1_fake").is_dir()
                                    for p in hit.iterdir() if p.is_dir()):
                split_dir = hit
                break
        if split_dir is None:
            raise FileNotFoundError(f"wildrf: khong thay split '{WILDRF_SPLIT}' duoi {root}")
        out = {}
        for plat in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            out[plat.name] = ([(p, 0) for p in list_images(plat / "0_real")] +
                              [(p, 1) for p in list_images(plat / "1_fake")])
        return out

    if name == "bfree_viral_images":
        root = optional_dir(BFREE_VIRAL_DIR)
        if root is None:
            raise FileNotFoundError("bfree_viral_images: chua dien BFREE_VIRAL_DIR o cell CONFIG-PATHS")
        base = _find_child_with(root, ["FAKE", "REAL"])
        if base is None:
            raise FileNotFoundError(f"bfree_viral_images: khong thay cap FAKE/REAL duoi {root}")
        return {"bfree_viral_images": ([(p, 0) for p in list_images(base / "REAL")] +
                                       [(p, 1) for p in list_images(base / "FAKE")])}

    raise ValueError(name)


def build_synthwildx_items():
    """synthwildx luu dang Arrow -> doc lazy giong ai_genbench, KHONG giai nen ra dia."""
    root = optional_dir(SYNTHWILDX_DIR)
    if root is None:
        raise FileNotFoundError("synthwildx: chua dien SYNTHWILDX_DIR o cell CONFIG-PATHS")
    from datasets import load_from_disk, concatenate_datasets

    obj = load_from_disk(str(root))
    ds = concatenate_datasets([obj[k] for k in obj.keys()]) if hasattr(obj, "keys") else obj
    img_col = next((c for c, f in ds.features.items() if f.__class__.__name__ == "Image"), None)
    label_col = next((c for c in ds.features if c.lower() in {"label", "target", "is_fake", "y", "class"}), None)
    if img_col is None or label_col is None:
        raise RuntimeError(f"synthwildx: khong nhan dien duoc cot anh/nhan. features={list(ds.features)}")

    names = list(getattr(ds.features[label_col], "names", []) or [])
    fake_ids = {i for i, n in enumerate(names) if "fake" in str(n).lower()} if names else None

    def to_y(v):
        v = v.item() if hasattr(v, "item") else v
        if fake_ids is not None:
            return 1 if int(v) in fake_ids else 0
        if isinstance(v, (int, np.integer)):
            return 0 if int(v) == 0 else 1
        return 0 if "real" in str(v).lower() else 1

    labels = ds.select_columns([label_col]).to_pandas()[label_col]
    items = [(i, to_y(v)) for i, v in enumerate(labels)]
    logger.info(f"synthwildx: n={len(items)} real={sum(1 for _, y in items if y == 0)} "
                f"fake={sum(1 for _, y in items if y == 1)}")
    return ArrowSource(ds, img_col), items

# %% ---

import pandas as pd

OUT_CSV = WORK / f"eval_results_{MODEL_ID}.csv"
rows = []


def save_partial():
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    logger.info(f"checkpoint ghi -> {OUT_CSV} ({len(rows)} dong)")


def _row(benchmark, generator, condition, metrics):
    rows.append({"model_id": MODEL_ID, "benchmark": benchmark, "generator": generator,
                 "condition": condition, **metrics})


# ------------------------------- ai_genbench -------------------------------
def run_ai_genbench():
    global CALIB_THRESHOLD, CALIB_SOURCE
    if not (BENCHMARKS_ENABLED.get("ai_genbench") and budget_ok()):
        return
    logger.info("=== ai_genbench ===")
    try:
        source, real_items, groups = build_ai_genbench_groups()
    except Exception as e:
        logger.error(f"[ai_genbench] BO QUA: {e}")
        return

    # Pool REAL score DUNG 1 LAN, dung lai cho moi generator (khong doi so voi truoc).
    yr_true, yr_prob = score_items(real_items, source=source, tag="ai_genbench/real")
    logger.info(f"ai_genbench: real pool scored n={len(yr_true)} mean_p_fake={np.mean(yr_prob):.4f}")

    # Buoc 1: score TUNG generator va LUU LAI (y_true, y_prob) trong bo nho — chua ghi row/CSV
    # ngay, vi can gop het pool nay lai truoc de tim CALIB_THRESHOLD (buoc 2), roi moi ghi row
    # o buoc 3 SU DUNG LAI score da co (khong rescoring lan 2, tranh ton gap doi thoi gian GPU).
    per_gen_scores = {}
    all_true, all_prob = list(yr_true), list(yr_prob)
    for gi, (gen, items) in enumerate(sorted(groups.items()), 1):
        if not budget_ok():
            logger.warning("ai_genbench: het ngan sach thoi gian, dung o generator ke tiep")
            break
        yf_true, yf_prob = score_items(items, source=source, tag=f"ai_genbench/{gen}")
        per_gen_scores[gen] = (yf_true, yf_prob)
        all_true += yf_true
        all_prob += yf_prob

    # Buoc 2: hieu chinh nguong bang Youden's J tren TOAN BO ai_genbench da chay (real pool +
    # tat ca generator) — day la calibration set lon nhat/da dang nhat hien co. CALIB_THRESHOLD
    # la bien GLOBAL nen moi loi goi compute_all_metrics() SAU DAY (wildrf, bfree_viral_images,
    # synthwildx, realdeg...) se TU DONG dung chung nguong nay, khong can sua them noi khac.
    if len(all_true) > len(yr_true):
        CALIB_THRESHOLD = find_optimal_threshold(all_true, all_prob, method="youden")
        CALIB_SOURCE = f"Youden J tren ai_genbench pooled (n={len(all_true)})"
        logger.info(f"[calibration] {CALIB_SOURCE}: CALIB_THRESHOLD={CALIB_THRESHOLD:.4f} "
                    f"(nguong cu = 0.5)")
    else:
        logger.warning("[calibration] khong du du lieu ai_genbench de hieu chinh nguong — "
                       f"giu CALIB_THRESHOLD={CALIB_THRESHOLD} (mac dinh)")

    # Buoc 3: ghi row cho TUNG generator, dung LAI score da luu o buoc 1 (khong goi score_items
    # lai lan nua) — moi row gio co ca 'bacc' (nguong 0.5, de so sanh voi ket qua truoc do) va
    # 'bacc_calib' + 'thr_calib' (nguong da hieu chinh).
    for gi, gen in enumerate(sorted(per_gen_scores), 1):
        yf_true, yf_prob = per_gen_scores[gen]
        m = compute_all_metrics(yr_true + yf_true, yr_prob + yf_prob)
        _row("ai_genbench", gen, "clean", m)
        logger.info(f"  [{gi}/{len(per_gen_scores)}] {gen:<28} n={m['n']:<5} auc={m['auc']:.4f} "
                    f"bacc@0.5={m['bacc']:.2f} bacc@calib={m['bacc_calib']:.2f}")
        save_partial()

    if len(all_true) > len(yr_true):
        _row("ai_genbench", "ALL", "clean", compute_all_metrics(all_true, all_prob))
        save_partial()
    AIGB_CACHE["last"] = (source, real_items, groups)


# ------------------------- cac bo in-the-wild (anh roi) -------------------------
def run_static_benchmark(name):
    if not (BENCHMARKS_ENABLED.get(name) and budget_ok()):
        return
    logger.info(f"=== {name} ===")
    try:
        groups = build_benchmark_filelist(name)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"[{name}] BO QUA: {e}")
        return
    for gname, items in groups.items():
        if not budget_ok():
            break
        m = eval_items(items, tag=f"{name}/{gname}")
        _row(name, gname, "clean", m)
        logger.info(f"  {gname:<20} n={m['n']:<5} auc={m['auc']:.4f} bacc={m['bacc']:.2f}")
    save_partial()


def run_synthwildx():
    if not (BENCHMARKS_ENABLED.get("synthwildx") and budget_ok()):
        return
    logger.info("=== synthwildx ===")
    try:
        source, items = build_synthwildx_items()
    except Exception as e:
        logger.error(f"[synthwildx] BO QUA: {e}")
        return
    m = eval_items(items, source=source, tag="synthwildx")
    _row("synthwildx", "synthwildx", "clean", m)
    logger.info(f"  synthwildx n={m['n']} auc={m['auc']:.4f} bacc={m['bacc']:.2f}")
    save_partial()


# ------------------------------ RealDeg-style ------------------------------
def run_realdeg(name, items, source=None, max_images=600):
    if not (BENCHMARKS_ENABLED.get("realdeg_bench_style") and budget_ok()):
        return
    logger.info(f"=== realdeg_bench_style on {name} (n<={max_images}) ===")
    rng_np = np.random.default_rng(AIGENBENCH_SEED)
    if max_images and len(items) > max_images:
        sel = rng_np.choice(len(items), max_images, replace=False)
        items = [items[i] for i in sorted(sel)]
    rng = random.Random(AIGENBENCH_SEED)
    for cond_name, cond_val in realdeg_conditions():
        if not budget_ok():
            break
        if cond_name == "clean":
            m = eval_items(items, source=source, tag=f"realdeg::{name}/{cond_name}")
        elif cond_name.startswith("single_"):
            m = eval_items(items, source=source, degrade_op=cond_val,
                           degrade_strength=rng.choice(DEGRADE_STRENGTHS[cond_val]),
                           tag=f"realdeg::{name}/{cond_name}")
        else:
            m = eval_items(items, source=source, compound_n=cond_val, rng=rng,
                           tag=f"realdeg::{name}/{cond_name}")
        _row(f"realdeg::{name}", "ALL", cond_name, m)
        logger.info(f"  {cond_name:<18} auc={m['auc']:.4f} bacc={m['bacc']:.2f}")
        save_partial()

# %% ---

# ---- Bo chinh (da loai synthbuster_original / genimage_val vi trung ai_genbench) ----
run_ai_genbench()
run_static_benchmark("wildrf")
run_synthwildx()
run_static_benchmark("bfree_viral_images")

# ---- RealDeg-Bench-style ----
if BENCHMARKS_ENABLED.get("realdeg_bench_style"):
    if "last" in AIGB_CACHE:
        src, real_items, groups = AIGB_CACHE["last"]
        pooled = list(real_items) + [it for v in groups.values() for it in v]
        run_realdeg("ai_genbench", pooled, source=src, max_images=600)
    for bench in ["wildrf", "bfree_viral_images"]:
        if not BENCHMARKS_ENABLED.get(bench):
            continue
        try:
            groups = build_benchmark_filelist(bench)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"[realdeg::{bench}] BO QUA: {e}")
            continue
        run_realdeg(bench, [it for v in groups.values() for it in v], max_images=600)

save_partial()
logger.info(f"HOAN TAT - {len(rows)} dong ket qua tai {OUT_CSV}")
logger.info(f"Thoi gian con lai: {time_left_hours():.2f}h")

# %% ---

# Tom tat nhanh
df = pd.DataFrame(rows)
if len(df):
    clean = df[df.condition == "clean"]
    cols = ["benchmark", "generator", "n", "auc", "bacc", "bacc_calib", "thr_calib", "ece", "nll"]
    cols = [c for c in cols if c in clean.columns]
    print(clean[cols].to_string(index=False))
    print(f"\n[calibration] nguon nguong: {CALIB_SOURCE} | CALIB_THRESHOLD hien tai = {CALIB_THRESHOLD:.4f}")
else:
    print("Chua co ket qua nao - xem log phia tren de biet benchmark nao bi skip.")
