from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import json
import time
import zlib
import traceback
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

from arc_sched import Pacer

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


def _env(name, default, cast):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default


# Defaults reproduce the public NVARC baseline (128 TTT sequences, 16 views, DFS p>=0.2).
CFG = dict(
    model_path       = _env("ARC_MODEL_PATH", "", str),
    out_dir          = _env("ARC_OUT_DIR", "/kaggle/inference_outputs", str),
    telemetry        = _env("ARC_TELEMETRY", "/kaggle/working/telemetry.jsonl", str),
    min_prob         = _env("ARC_MIN_PROB", 0.2, float),
    extra_min_prob   = _env("ARC_EXTRA_MIN_PROB", 0.12, float),
    window_per_view  = _env("ARC_WINDOW_PER_VIEW", 135.0, float),   # baseline: 540 s per 4-view batch
    max_rows         = _env("ARC_MAX_ROWS", 8, int),
    rows8_max_ctx    = _env("ARC_ROWS8_MAX_CTX", 3600, int),
    merge_lora       = _env("ARC_MERGE_LORA", 0, int),   # off: bf16 rounding of W + B@A erodes the TTT update
    bench            = _env("ARC_BENCH", 0, int),
)


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


# Minimal performance patch: preserve the baseline beam set and ranking, but transfer
# only the 12 ARC-token NLL values to CPU instead of every Qwen vocabulary logit.
_ARC_TOKEN_ID_CACHE = {}


def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


DFS_STEPS = [0]


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, window) -> dict:

    n = logits.size(0)

    # Algebraically identical to: scores - logits.float().cpu().log_softmax(-1),
    # restricted to the same ARC_TOKENS used by the baseline DFS loop.
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0])

    while time.time() - start_time < window and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        DFS_STEPS[0] += 1
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            window=window,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, window):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
        window=window,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    # Keep logits on GPU and gather only the target-token scores. KV cache is not
    # consumed by teacher-forced scoring, so disabling it removes redundant writes.
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result


def stable_seed(key, offset=0):
    # The baseline used hash(bk), which is salted per process (PYTHONHASHSEED unset);
    # crc32 keeps the 8 rescoring views identical across workers and reruns.
    return (zlib.crc32(key.encode("utf-8")) + offset) % (1024 ** 2)


def resolve_model_dir():
    if CFG["model_path"] and os.path.isfile(os.path.join(CFG["model_path"], "config.json")):
        return CFG["model_path"]
    candidates = [
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    import glob
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if "grids15" in d and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError("qwen3_4b_grids15_sft139 model directory not found under /kaggle/input")


# ---------------------------------------------------------------------------
# Inference-time LoRA merge.
# Unsloth's fast inference path applies the r=256 adapter as two extra matmuls per
# projection (14 per layer). After TTT we fold B@A into the frozen base weights and mark
# the layer merged (unsloth's get_lora_parameters and PEFT both skip the adapter then).
# The original bf16 base weights are restored bit-exactly from a CPU snapshot before
# the next task trains, so no rounding drift accumulates across tasks.
# ---------------------------------------------------------------------------
class LoraMerger:

    def __init__(self, model, snapshot_path=None):
        from peft.tuners.lora import LoraLayer
        self.layers = []
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and isinstance(getattr(module, "base_layer", None), torch.nn.Linear):
                if "default" in module.lora_A and isinstance(getattr(module, "merged_adapters", None), list):
                    self.layers.append((name, module))
        if snapshot_path is None:
            self.snapshot = {
                name: module.base_layer.weight.detach().to("cpu", copy=True)
                for name, module in self.layers
            }
        else:
            self.snapshot = self._shared_snapshot(snapshot_path)
            # the file was written by whichever worker got there first: check it is ours
            for name, module in self.layers[:: max(1, len(self.layers) // 8)]:
                if not torch.equal(module.base_layer.weight.detach().cpu(), self.snapshot[name]):
                    raise RuntimeError(f"base-weight snapshot mismatch at {name}")
        self.merged = False

    def _shared_snapshot(self, path):
        """One copy of the frozen base weights for all workers: written once to local disk,
        memory-mapped by every worker so the page cache is shared (4 private copies would
        cost ~29 GB of host RAM)."""
        lock = path + ".lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            owner = True
        except FileExistsError:
            owner = False
        if owner:
            state = {name: module.base_layer.weight.detach().to("cpu", copy=True) for name, module in self.layers}
            torch.save(state, path + ".tmp")
            del state
            os.replace(path + ".tmp", path)
        else:
            t0 = time.time()
            while not os.path.exists(path):
                if time.time() - t0 > 300:
                    raise TimeoutError("shared base-weight snapshot was never written")
                time.sleep(2)
        return torch.load(path, mmap=True, weights_only=True)

    @torch.no_grad()
    def merge(self):
        self.merged = True  # set first: restore() must run even if merging stops half-way
        for name, module in self.layers:
            W = module.base_layer.weight
            A = module.lora_A["default"].weight.float()
            B = module.lora_B["default"].weight.float()
            s = module.scaling["default"]
            W.copy_((W.float() + (B @ A) * s).to(W.dtype))
            module.merged_adapters.append("default")

    @torch.no_grad()
    def restore(self):
        if not self.merged:
            return
        for name, module in self.layers:
            module.base_layer.weight.copy_(self.snapshot[name], non_blocking=False)
            module.merged_adapters.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.merged = False


def view_geometry(subkey):
    ops = subkey.split(".")[1:]
    transposed = len(ops) > 0 and ops[0] == "transpose"
    rot = sum(1 for op in ops if op == "rot90")
    return transposed, rot


def make_batches(eval_ds, formatter, tokenizer, max_rows, rows8_max_ctx):
    """Group decoded views into DFS batches.

    Turbo DFS shares one position id across rows, so every row of a batch must have the
    same prompt length; views are grouped by their exact token count (shape class
    {id, rot180, T.rot90, T.rot270} vs {rot90, rot270, T, T.rot180}, minus any
    examples cut_to_len dropped). Short prompts decode 8 rows per batch, long ones 4.
    Batches are interleaved across test outputs so a time cap leaves every output
    with views from both shape classes, as in the baseline order."""
    per_test = defaultdict(lambda: defaultdict(list))
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        tokens = tokenizer.encode(eval_ds.get(subkey, formatter)["input"])
        per_test[test_id][len(tokens)].append((subkey, tokens))
    chunks_per_test = {}
    for test_id, groups in per_test.items():
        chunks = []
        # largest group first == the class containing the identity view in normal cases
        for length, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            rows = max_rows if length <= rows8_max_ctx else min(4, max_rows)
            for i in range(0, len(items), rows):
                chunks.append(items[i:i + rows])
        # alternate shape classes: A1, B1, A2, B2 ...
        half = defaultdict(list)
        for c in chunks:
            half[len(c[0][1])].append(c)
        ordered = []
        lists = list(half.values())
        for i in range(max(len(l) for l in lists)):
            for l in lists:
                if i < len(l):
                    ordered.append(l[i])
        chunks_per_test[test_id] = ordered
    batches = []
    n = max(len(c) for c in chunks_per_test.values())
    for i in range(n):
        for test_id in sorted(chunks_per_test):
            if i < len(chunks_per_test[test_id]):
                batches.append(chunks_per_test[test_id][i])
    return batches


def output_confident(results_by_bk, bk, n_views):
    """True if one candidate clearly dominates the views decoded so far."""
    counts = defaultdict(int)
    for sol in results_by_bk.get(bk, []):
        counts[sol] += 1
    if not counts:
        return False
    top = sorted(counts.values(), reverse=True)
    c1 = top[0]
    c2 = top[1] if len(top) > 1 else 0
    return c1 >= max(6, n_views // 2) and c1 >= 2 * c2 + 2


def tel(path, rec):
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def worker(rank, end_time, shared, n_workers, test_path):

    # turbo_dfs recurses once per generated token (up to 931 for a 30x30 grid)
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 6000))

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    bench_pending = bool(CFG["bench"])

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolve_model_dir(),
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    merger = None
    if CFG["merge_lora"]:
        try:
            merger = LoraMerger(model, snapshot_path=os.getenv("ARC_SNAPSHOT", "/tmp/arc_base_weights.pt"))
            print(f"[Rank {rank}] LoRA merge enabled over {len(merger.layers)} layers")
            if not merger.layers:
                merger = None
        except Exception as e:
            print(f"[Rank {rank}] LoRA merge disabled: {e!r}")
            merger = None
    merge_checked = False

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = CFG["out_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    pacer = Pacer(shared, rank, n_workers, end_time, arc_test_set.queries)

    def bench_dfs(items, label):
        tokens = [t for _, t in items]
        torch.cuda.synchronize()
        t0 = time.time()
        steps0 = DFS_STEPS[0]
        with torch.inference_mode():
            res = inference_turbo_dfs(model, tokens, max_new_tokens, -np.log(CFG["min_prob"]), end_time, 600.0)
        torch.cuda.synchronize()
        dt = time.time() - t0
        steps = DFS_STEPS[0] - steps0
        beams = sorted((b, tuple(tk)) for _, bs in res for b, tk in [(round(x[0], 3), x[1]) for x in bs])
        print(f"[Rank {rank}] BENCH {label}: rows {len(items)} ctx {len(tokens[0])} time {dt:.1f}s "
              f"steps {steps} ms/step {1000 * dt / max(steps, 1):.1f} beams {len(beams)}")
        return dict(label=label, rows=len(items), ctx=len(tokens[0]), t=round(dt, 2), steps=steps), beams

    def run_task(key, plan):
        nonlocal model, merge_checked, merger, bench_pending

        start_time = time.time()
        deadline = min(start_time + plan["budget"], end_time)
        info = dict(task=key, rank=rank, pass_name=plan["pass_name"], level=plan["level"],
                    budget=round(plan["budget"], 1), est=round(plan["est"], 1), kappa=round(plan["kappa"], 3))

        torch.cuda.reset_peak_memory_stats()

        set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])

        train_ds = puzzle_ds.augment(n=plan["n_train_aug"], shfl_keys=True, seed=plan["train_seed"])
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        train_args_task = dict(train_args, seed=plan["lora_seed"])
        if plan["lora_seed"] != 42:
            torch.manual_seed(plan["lora_seed"])

        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args_task),
            )

            stats = trainer.train()

            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

            del trainer

        model = FastLanguageModel.for_inference(model)

        gc.collect()
        torch.cuda.empty_cache()

        info["train_s"] = round(time.time() - start_time, 1)
        info["train_steps"] = int(stats.global_step)
        info["train_loss"] = round(float(stats.training_loss), 5)
        info["train_mem_mb"] = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] trained {key} ({plan['pass_name']}, level {plan['level']}, "
              f"{stats.global_step} steps) in {info['train_s']}s, loss {info['train_loss']}")

        torch.cuda.reset_peak_memory_stats()

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        eval_ds = puzzle_ds_multi.augment(n=2, seed=plan["eval_seed"])
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

        batches = make_batches(eval_ds, formatter, tokenizer, CFG["max_rows"], CFG["rows8_max_ctx"])

        bench = []
        if bench_pending and merger is not None and len(batches[0]) >= 8:
            bench_pending = False
            r, beams_unmerged = bench_dfs(batches[0], "unmerged")
            bench.append(r)

        # --- merge the adapter for decoding (verified once per worker) ---
        if merger is not None:
            try:
                probe_q = [eval_ds.get(batches[0][0][0], formatter)["input"]]
                probe_a = [formatter.fmt_reply([puzzle_ds_multi.queries[batches[0][0][0].split(".")[0]]["test"][0]["input"]])]
                if not merge_checked:
                    with torch.inference_mode():
                        ref = calc_scores(probe_q, probe_a, tokenizer, model)[0]
                merger.merge()
                if not merge_checked:
                    with torch.inference_mode():
                        got = calc_scores(probe_q, probe_a, tokenizer, model)[0]
                    ok = abs(got - ref) <= 0.03 * abs(ref) + 1.0
                    print(f"[Rank {rank}] merge self-test: unmerged NLL {ref:.4f} merged NLL {got:.4f} -> {'OK' if ok else 'MISMATCH'}")
                    info["merge_check"] = [round(ref, 4), round(got, 4), ok]
                    merge_checked = True
                    if not ok:
                        merger.restore()
                        merger = None
            except Exception as e:
                print(f"[Rank {rank}] merge failed, continuing unmerged: {e!r}")
                try:
                    merger.restore()
                except Exception:
                    pass
                merger = None

        if bench and merger is not None and merger.merged:
            r, beams_merged = bench_dfs(batches[0], "merged")
            bench.append(r)
            r["same_beams"] = [t for _, t in beams_merged] == [t for _, t in beams_unmerged]
            r, _ = bench_dfs(batches[0][:4], "merged-4rows")
            bench.append(r)
            info["bench"] = bench

        known_scores = {}
        sols_by_bk = defaultdict(list)
        n_views_done = defaultdict(int)
        dfs_log = []

        def decode_batch(items, max_score, suffix):
            subkeys = [s for s, _ in items]
            tokens = [t for _, t in items]
            window = CFG["window_per_view"] * len(items)
            t0 = time.time()
            steps0 = DFS_STEPS[0]
            try:
                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, min(deadline, end_time), window)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if len(items) == 1:
                    print(f"[Rank {rank}] OOM on single view {subkeys[0]}, skipped")
                    return
                print(f"[Rank {rank}] OOM with {len(items)} rows, splitting")
                h = len(items) // 2
                decode_batch(items[:h], max_score, suffix)
                decode_batch(items[h:], max_score, suffix)
                return
            dt = time.time() - t0
            steps = DFS_STEPS[0] - steps0
            dfs_log.append((len(items), len(tokens[0]), round(dt, 1), steps, round(1000 * dt / max(steps, 1), 1)))
            for subkey in subkeys:
                n_views_done[subkey.split(".")[0]] += 1

            for subkey_id, scored_beams in dfs_result:

                subkey = subkeys[subkey_id]
                bk = subkey.split(".")[0]
                decoded_result = []

                for beam_score, toks in scored_beams:

                    array = formatter.convert_tokens_to_array(toks)
                    if array is None:
                        continue

                    solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                    grid_id = (bk, tuple(map(tuple, solution)))

                    if grid_id in known_scores:
                        augmented_scores = known_scores[grid_id]
                    else:
                        aug_dataset = ArcDataset(
                            keys=[bk],
                            queries={bk: puzzle_ds_multi.queries.get(bk)},
                            replies={bk: [solution.tolist()]},
                        )
                        aug_dataset = aug_dataset.augment(seed=stable_seed(bk))
                        aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                        aug_queries = []
                        aug_answers = []
                        for augmented_sample in aug_dataset.as_list(formatter):
                            aug_queries.append(augmented_sample["input"])
                            aug_answers.append(augmented_sample["reply"])
                        augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                        augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                        augmented_scores = augmented_scores1 + augmented_scores2
                        known_scores[grid_id] = augmented_scores

                    decoded_result.append({
                        "beam_score": beam_score,
                        "score_aug": augmented_scores,
                        "solution": solution,
                    })
                    sols_by_bk[bk].append(grid_id[1])

                if len(decoded_result):
                    with bz2.BZ2File(os.path.join(dir_outputs, subkey + suffix), "w") as f:
                        pickle.dump(decoded_result, f)

        timed_out = False
        with torch.inference_mode():

            for items in batches:
                if time.time() > deadline - 5 or time.time() > end_time - 30:
                    timed_out = True
                    break
                decode_batch(items, -np.log(CFG["min_prob"]), plan["suffix"])

            info["base_s"] = round(time.time() - start_time, 1)

            # --- adaptive extra views for outputs without a dominant candidate ---
            extra_done = 0
            if plan["extra_views"] and not timed_out:
                bks = sorted(puzzle_ds_multi.keys)
                todo = [bk for bk in bks if not output_confident(sols_by_bk, bk, n_views_done[bk])]
                if todo:
                    extra_ds = puzzle_ds_multi.change_keys(todo).augment(n=1, seed=plan["eval_seed"] + 1000)
                    extra_ds = extra_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                    extra_batches = make_batches(extra_ds, formatter, tokenizer, CFG["max_rows"], CFG["rows8_max_ctx"])
                    for items in extra_batches:
                        if time.time() > deadline - 5 or time.time() > end_time - 30:
                            break
                        decode_batch(items, -np.log(CFG["extra_min_prob"]), plan["suffix"] + ".x")
                        extra_done += len(items)
                info["extra_outputs"] = len(todo)
            info["extra_views"] = extra_done

        timed_out = timed_out or time.time() >= deadline - 2

        if merger is not None:
            merger.restore()

        # uncertainty summary for the tail pass
        unc = {}
        for bk in sorted(puzzle_ds_multi.keys):
            counts = defaultdict(int)
            for s in sols_by_bk.get(bk, []):
                counts[s] += 1
            top = sorted(counts.values(), reverse=True)
            unc[bk] = dict(n_unique=len(counts), c1=top[0] if top else 0,
                           c2=top[1] if len(top) > 1 else 0, views=n_views_done[bk])
        info["unc"] = unc
        info["dfs"] = dfs_log
        info["infer_mem_mb"] = torch.cuda.max_memory_allocated() // 1024**2
        info["elapsed"] = round(time.time() - start_time, 1)
        info["timed_out"] = timed_out
        tel(CFG["telemetry"], info)
        print(f"[Rank {rank}] finished {key} ({plan['pass_name']}) in {info['elapsed']}s "
              f"[train {info['train_s']}s, base views {info['base_s']}s, extra views {extra_done}, "
              f"budget {plan['budget']:.0f}s, dfs {dfs_log}]")
        return info

    while True:
        plan = pacer.next_task()
        if plan is None:
            break
        key = plan["task"]
        t0 = time.time()
        try:
            info = run_task(key, plan)
            pacer.done(plan, time.time() - t0, info)
        except Exception as e:
            print(f"[Rank {rank}] ERROR on {key}: {e!r}")
            traceback.print_exc()
            pacer.done(plan, time.time() - t0, None)
            try:
                if merger is not None:
                    merger.restore()
            except Exception as e2:
                print(f"[Rank {rank}] restore after error failed: {e2!r}; disabling merge")
                merger = None
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[Rank {rank}] no more work")
