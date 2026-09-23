# %%
# ARC search comparison: 10-20 puzzle end-to-end experiment
#
# This is valid Python: run it as a script, paste it into one Kaggle cell, or
# paste the # %% sections into separate cells. No notebook cell magics are used.
#
# It preserves the original task-time LoRA training. After training each puzzle,
# DFS, A* h=0, beam-4, true greedy, and top-p run on the same fixed augmented
# prompt. The production 16-view inference/rescoring stage is intentionally
# skipped so this pilot isolates the search algorithm and finishes much faster.

import os
import subprocess
import sys
import time
from pathlib import Path

NUM_PUZZLES = 10          # Set to 20 for the larger pilot.
PUZZLE_SEED = 42          # Reproducible random evaluation subset.
NUM_WORKERS = 4           # Uses the original 4 x L4 setup when available.
TOTAL_RUNTIME_HOURS = 6   # Safety ceiling, not an expected runtime.
SAFETY_BUFFER_SECONDS = 10 * 60
OUTPUT_DIR = "/kaggle/working/search_compare_outputs"

assert 10 <= NUM_PUZZLES <= 20, "Use 10-20 puzzles for this pilot."

global_end_time = (
    time.time()
    + TOTAL_RUNTIME_HOURS * 3600
    - SAFETY_BUFFER_SECONDS
)

print("Puzzles:", NUM_PUZZLES)
print("Subset seed:", PUZZLE_SEED)
print("Global end time:", time.ctime(global_end_time))


# %%
# Preserve the baseline environment workaround.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"],
    check=False,
)


# %%
# Write the dataset/formatting module.
Path("arc_loader.py").write_text(
    r'''import json
import numpy as np
from transformers import AutoTokenizer


def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()

def is_valid_solution(guess):
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))


class QwenFormatter:

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except:
            pass
        return None


class ArcDataset:

    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if   op=='rot90':              a = np.rot90(a, k=3)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file, keys=None):
        with open(queries_file) as f:
            queries = f.read()
        return cls(
            queries=json.loads(queries),
            is_orig=True,
            keys=keys,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: QwenFormatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: QwenFormatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name])) for ds in datasets)

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            reply = temp_ds.replies.get(key)
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None:
                    temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)
    
    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n=1, shfl_keys=False, seed=42):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target_dict)]):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score''',
    encoding="utf-8",
)
print("Wrote arc_loader.py")


# %%
# Write the task-time training and search module.
Path("arc_solver.py").write_text(
    r'''from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
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

# Search experiment settings.  DFS and A* get the same wall-clock budget.
SEARCH_COMPARE_K = 4
SEARCH_BEAM_WIDTH = 4
SEARCH_TOP_P = 0.95
SEARCH_TEMPERATURE = 1.0
SEARCH_MAX_SECONDS = 180.0
SEARCH_ASTAR_MAX_EXPANSIONS = 2500


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


def turbo_dfs(
    model,
    logits,
    max_new_tokens,
    max_score,
    scores,
    pos,
    cache,
    start_time,
    end_time,
    max_seconds=SEARCH_MAX_SECONDS,
) -> dict:

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
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < max_seconds and time.time() < end_time:

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
            max_seconds=max_seconds,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
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
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


# ============================================================
# SEARCH-COMPARISON DECODERS
# ============================================================

def _search_legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def _search_arc_step_nll(logits):
    """Full-vocabulary token NLL, restricted to the 12 valid ARC tokens."""
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    return (
        torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    )[0]


@torch.inference_mode()
def inference_astar_h0(
    model,
    prefix_tokens,
    max_new_tokens,
    max_score,
    n_goals=SEARCH_COMPARE_K,
    max_expansions=SEARCH_ASTAR_MAX_EXPANSIONS,
    max_seconds=SEARCH_MAX_SECONDS,
    hard_end_time=None,
):
    """A* with h=0, i.e. uniform-cost search over cumulative token NLL."""
    import heapq
    import itertools

    start_time = time.time()
    input_ids = torch.tensor(
        prefix_tokens,
        device=model.device,
        dtype=torch.long,
    ).view(1, -1)

    root = model(input_ids=input_ids, return_dict=True, use_cache=True)
    root_logits = root.logits[:, -1]
    root_cache = _search_legacy_cache(root.past_key_values)
    root_pos = input_ids.size(1)

    counter = itertools.count()
    frontier = [(0.0, next(counter), tuple())]
    goals = []
    expansions = 0

    active_seq = tuple()
    active_cache = root_cache
    active_logits = root_logits
    stopped_by_budget = False

    while frontier and len(goals) < n_goals:
        now = time.time()
        if (
            expansions >= max_expansions
            or now - start_time >= max_seconds
            or (hard_end_time is not None and now >= hard_end_time)
        ):
            stopped_by_budget = True
            break

        score, _, seq = heapq.heappop(frontier)
        if score >= max_score:
            break

        if seq and seq[-1] == EOS_ID:
            goals.append((score, list(seq)))
            continue

        if len(seq) >= max_new_tokens:
            continue

        if seq == active_seq:
            logits = active_logits
        elif len(seq) == len(active_seq) + 1 and seq[:-1] == active_seq:
            token = seq[-1]
            out = model(
                input_ids=torch.tensor(
                    [[token]], device=model.device, dtype=torch.long
                ),
                position_ids=torch.tensor(
                    [[root_pos + len(active_seq)]],
                    device=model.device,
                    dtype=torch.long,
                ),
                past_key_values=active_cache,
                return_dict=True,
                use_cache=True,
            )
            active_seq = seq
            active_cache = _search_legacy_cache(out.past_key_values)
            active_logits = out.logits[:, -1]
            logits = active_logits
        else:
            seq_tensor = torch.tensor(
                [list(seq)], device=model.device, dtype=torch.long
            )
            position_ids = torch.arange(
                root_pos,
                root_pos + len(seq),
                device=model.device,
                dtype=torch.long,
            ).view(1, -1)
            out = model(
                input_ids=seq_tensor,
                position_ids=position_ids,
                past_key_values=root_cache,
                return_dict=True,
                use_cache=True,
            )
            active_seq = seq
            active_cache = _search_legacy_cache(out.past_key_values)
            active_logits = out.logits[:, -1]
            logits = active_logits

        step_nll = _search_arc_step_nll(logits).detach().cpu().tolist()
        expansions += 1

        for token_id, delta in zip(ARC_TOKENS, step_nll):
            new_score = score + float(delta)
            if new_score >= max_score:
                continue
            child = seq + (token_id,)
            if token_id == EOS_ID or len(child) < max_new_tokens:
                heapq.heappush(
                    frontier,
                    (new_score, next(counter), child),
                )

    return {
        "beams": goals,
        "expansions": expansions,
        "frontier_size": len(frontier),
        "stopped_by_budget": stopped_by_budget,
        "seconds": time.time() - start_time,
    }


def _trim_search_suffix(sequence, prefix_len):
    suffix = [int(x) for x in sequence[prefix_len:]]
    if EOS_ID not in suffix:
        return None
    return suffix[: suffix.index(EOS_ID) + 1]


@torch.inference_mode()
def inference_hf_search(
    model,
    prefix_tokens,
    method,
    max_new_tokens,
    k=SEARCH_COMPARE_K,
    seed=0,
):
    from transformers import LogitsProcessor, LogitsProcessorList

    class ArcOnlyLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            token_ids = _arc_token_ids(scores.device)
            allowed = scores.index_select(-1, token_ids).clone()
            scores.fill_(torch.finfo(scores.dtype).min)
            scores.index_copy_(-1, token_ids, allowed)
            return scores

    input_ids = torch.tensor(
        prefix_tokens,
        device=model.device,
        dtype=torch.long,
    ).view(1, -1)

    kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        logits_processor=LogitsProcessorList([ArcOnlyLogitsProcessor()]),
        use_cache=True,
    )

    if method == "greedy":
        kwargs.update(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    elif method == "beam":
        kwargs.update(
            do_sample=False,
            num_beams=SEARCH_BEAM_WIDTH,
            num_return_sequences=min(k, SEARCH_BEAM_WIDTH),
            length_penalty=0.0,
            early_stopping=False,
        )
    elif method == "top_p":
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        kwargs.update(
            do_sample=True,
            num_beams=1,
            num_return_sequences=k,
            top_p=SEARCH_TOP_P,
            temperature=SEARCH_TEMPERATURE,
        )
    else:
        raise ValueError(method)

    start_time = time.time()
    sequences = model.generate(**kwargs)
    elapsed = time.time() - start_time

    suffixes = []
    for seq in sequences.detach().cpu().tolist():
        suffix = _trim_search_suffix(seq, len(prefix_tokens))
        if suffix is not None:
            suffixes.append(suffix)

    return {"suffixes": suffixes, "seconds": elapsed}


def _decode_search_solutions(
    scored_suffixes,
    formatter,
    puzzle_ds_multi,
    subkey,
    limit=SEARCH_COMPARE_K,
):
    results = []
    seen = set()

    for score, suffix in scored_suffixes:
        if not suffix or suffix[-1] != EOS_ID:
            continue
        array = formatter.convert_tokens_to_array(suffix)
        if array is None:
            continue
        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
        grid_id = tuple(map(tuple, solution))
        if grid_id in seen:
            continue
        seen.add(grid_id)
        results.append(
            {
                "solution": solution.tolist(),
                "score": None if score is None else float(score),
            }
        )
        if len(results) >= limit:
            break

    return results


def run_search_comparison(
    model,
    prefix_tokens,
    formatter,
    puzzle_ds_multi,
    subkey,
    max_new_tokens,
    max_score,
    end_time,
):
    """Run every decoder on the exact same fine-tuned model and prompt."""
    import zlib

    result = {
        "subkey": subkey,
        "basekey": subkey.split(".")[0],
        "k": SEARCH_COMPARE_K,
        "methods": {},
    }

    # True DFS, run by itself so its timing is directly comparable.
    try:
        dfs_t0 = time.time()
        dfs_batches = inference_turbo_dfs(
            model,
            [prefix_tokens],
            max_new_tokens,
            max_score,
            end_time,
        )
        dfs_seconds = time.time() - dfs_t0
        dfs_beams = next(
            (beams for batch_id, beams in dfs_batches if batch_id == 0),
            [],
        )
        result["methods"]["dfs"] = {
            "solutions": _decode_search_solutions(
                dfs_beams, formatter, puzzle_ds_multi, subkey
            ),
            "seconds": float(dfs_seconds),
            "num_raw_completed": len(dfs_beams),
        }
    except Exception as exc:
        result["methods"]["dfs"] = {
            "solutions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    for method in ("greedy", "beam", "top_p"):
        seed = zlib.crc32(f"{subkey}:{method}".encode("utf-8")) & 0xFFFFFFFF
        try:
            search = inference_hf_search(
                model=model,
                prefix_tokens=prefix_tokens,
                method=method,
                max_new_tokens=max_new_tokens,
                k=SEARCH_COMPARE_K,
                seed=seed,
            )
            result["methods"][method] = {
                "solutions": _decode_search_solutions(
                    [(None, x) for x in search["suffixes"]],
                    formatter,
                    puzzle_ds_multi,
                    subkey,
                ),
                "seconds": float(search["seconds"]),
                "num_raw_completed": len(search["suffixes"]),
            }
        except Exception as exc:
            result["methods"][method] = {
                "solutions": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    try:
        astar = inference_astar_h0(
            model=model,
            prefix_tokens=prefix_tokens,
            max_new_tokens=max_new_tokens,
            max_score=max_score,
            n_goals=SEARCH_COMPARE_K,
            hard_end_time=end_time,
        )
        result["methods"]["astar_h0"] = {
            "solutions": _decode_search_solutions(
                astar["beams"], formatter, puzzle_ds_multi, subkey
            ),
            "seconds": float(astar["seconds"]),
            "num_raw_completed": len(astar["beams"]),
            "expansions": astar["expansions"],
            "frontier_size": astar["frontier_size"],
            "stopped_by_budget": astar["stopped_by_budget"],
        }
    except Exception as exc:
        result["methods"]["astar_h0"] = {
            "solutions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    return result


def worker(rank, queue, end_time, output_dir):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

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
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
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

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    max_score = -np.log(0.2)

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    os.makedirs(output_dir, exist_ok=True)

    while True:

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break
        
        start_time = time.time()
        
        torch.cuda.reset_peak_memory_stats()

        set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])

        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            
            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args),
            )

            stats = trainer.train()

            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

            del trainer

        model = FastLanguageModel.for_inference(model)
        
        gc.collect()
        torch.cuda.empty_cache()
            
        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

        torch.cuda.reset_peak_memory_stats()
        
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        # Use one fixed augmented view for every test output.  This isolates the
        # decoder: every method receives the same prompt and fine-tuned model.
        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(
            formatter=formatter,
            name="input",
            max_len=max_seq_length - max_new_tokens,
        )

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0]
            test_id_to_subkeys[test_id].append(subkey)

        with torch.inference_mode():
            for basekey, subkeys in sorted(test_id_to_subkeys.items()):
                if time.time() >= end_time:
                    print(f"[Rank {rank}] global timeout before {basekey}")
                    break

                subkey = subkeys[0]
                sample = eval_ds.get(subkey, formatter)
                prefix_tokens = tokenizer.encode(sample["input"])

                print(f"[Rank {rank}] comparing searches for {subkey}")
                try:
                    compare_result = run_search_comparison(
                        model=model,
                        prefix_tokens=prefix_tokens,
                        formatter=formatter,
                        puzzle_ds_multi=puzzle_ds_multi,
                        subkey=subkey,
                        max_new_tokens=max_new_tokens,
                        max_score=max_score,
                        end_time=end_time,
                    )
                    output_path = os.path.join(output_dir, f"{basekey}.pbz2")
                    with bz2.BZ2File(output_path, "w") as f:
                        pickle.dump(compare_result, f)
                    print(f"[Rank {rank}] saved {output_path}")
                except Exception as exc:
                    print(
                        f"[Rank {rank}] comparison failed for {subkey}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        
        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")''',
    encoding="utf-8",
)
print("Wrote arc_solver.py")


# %%
# Write the multi-GPU launcher.
Path("starter.py").write_text(
    r'''
import os
import time
import json
import random
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, output_dir):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Prevent Unsloth initialization race.
    if rank > 0:

        while not os.path.exists(
            f"/kaggle/worker{rank-1}"
        ):
            time.sleep(5)

    from arc_solver import worker

    with open(
        f"/kaggle/worker{rank}",
        "w"
    ) as f:
        f.write("Ok")

    print(
        f"[Rank {rank}] start!"
    )

    worker(
        rank,
        queue,
        end_time,
        output_dir,
    )

    print(
        f"[Rank {rank}] done!"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--end-time",
        type=float,
        default=0.0
    )

    parser.add_argument("--num-puzzles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/search_compare_outputs",
    )

    args = parser.parse_args()

    rerun_mode = os.getenv(
        "KAGGLE_IS_COMPETITION_RERUN"
    )

    if rerun_mode:

        test_path = (
            "/kaggle/input/competitions/"
            "arc-prize-2026-arc-agi-2/"
            "arc-agi_test_challenges.json"
        )

    else:

        test_path = (
            "/kaggle/input/competitions/"
            "arc-prize-2026-arc-agi-2/"
            "arc-agi_evaluation_challenges.json"
        )

    with open(test_path, "r") as f:
        data = json.load(f)

    all_keys = sorted(data.keys())
    if not 1 <= args.num_puzzles <= len(all_keys):
        raise ValueError(
            f"--num-puzzles must be between 1 and {len(all_keys)}"
        )

    selected_keys = sorted(
        random.Random(args.seed).sample(all_keys, args.num_puzzles)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "selected_puzzles.json"), "w") as f:
        json.dump(
            {
                "num_puzzles": args.num_puzzles,
                "seed": args.seed,
                "keys": selected_keys,
            },
            f,
            indent=2,
        )

    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError("This experiment requires at least one CUDA GPU.")
    num_workers = min(args.num_workers, available_gpus)
    if num_workers != args.num_workers:
        print(
            f"Requested {args.num_workers} workers but found {available_gpus} GPUs; "
            f"using {num_workers} workers."
        )

    queue = mp.Manager().Queue()

    for key in selected_keys:

        queue.put(key)

    print(
        f"Queued {len(selected_keys)} puzzles: {selected_keys}"
    )

    # One sentinel per worker.
    for _ in range(num_workers):
        queue.put(None)

    mp.spawn(
        local_worker,
        args=(
            queue,
            args.end_time,
            args.output_dir,
        ),
        nprocs=num_workers,
    )''',
    encoding="utf-8",
)
print("Wrote starter.py")


# %%
# ============================================================
# RUN THE 10-20 PUZZLE EXPERIMENT
# ============================================================

import shutil

# Remove only this experiment's old outputs so reruns cannot mix results.
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for rank in range(NUM_WORKERS):
    marker = Path(f"/kaggle/worker{rank}")
    if marker.exists():
        marker.unlink()

run_env = os.environ.copy()
run_env.update(
    {
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "OMP_NUM_THREADS": "12",
    }
)

subprocess.run(
    [
        sys.executable,
        "starter.py",
        "--end-time",
        str(global_end_time),
        "--num-puzzles",
        str(NUM_PUZZLES),
        "--seed",
        str(PUZZLE_SEED),
        "--num-workers",
        str(NUM_WORKERS),
        "--output-dir",
        OUTPUT_DIR,
    ],
    env=run_env,
    check=True,
)


# %%
# ============================================================
# LOAD AND ANALYZE SEARCH-COMPARISON RESULTS
# ============================================================

import bz2
import json
import os
import pickle

import numpy as np
import pandas as pd

from arc_loader import ArcDataset


records = []
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if not filename.endswith(".pbz2"):
        continue
    with bz2.BZ2File(os.path.join(OUTPUT_DIR, filename), "r") as f:
        records.append(pickle.load(f))

with open(os.path.join(OUTPUT_DIR, "selected_puzzles.json"), "r") as f:
    selection = json.load(f)

print("Selected puzzle IDs:", len(selection["keys"]))

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
if rerun_mode:
    raise RuntimeError(
        "Ground-truth metrics are unavailable in competition rerun mode. "
        "Run this experiment against the evaluation split."
    )

evaluation = ArcDataset.from_file(
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
    "arc-agi_evaluation_challenges.json"
).load_replies(
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
    "arc-agi_evaluation_solutions.json"
)
evaluation_multi = evaluation.split_multi_replies()

expected_test_outputs = sum(
    len(evaluation.queries[key]["test"])
    for key in selection["keys"]
)
print("Expected test outputs:", expected_test_outputs)
print("Completed test outputs:", len(records))
print(
    "Run coverage:",
    f"{len(records) / expected_test_outputs:.1%}",
)

rows = []
for rec in records:
    basekey = rec["basekey"]
    ground_truth = np.asarray(evaluation_multi.replies[basekey][0])

    for method, info in rec["methods"].items():
        solutions = [np.asarray(x["solution"]) for x in info["solutions"]]
        hits = [np.array_equal(solution, ground_truth) for solution in solutions]
        gt_rank = hits.index(True) + 1 if any(hits) else np.nan

        rows.append(
            {
                "puzzle_id": basekey.rsplit("_", 1)[0],
                "test_output_id": basekey,
                "benchmark_subkey": rec["subkey"],
                "method": method,
                "top1_exact": bool(hits[0]) if hits else False,
                "recall_at_k": bool(any(hits)),
                "gt_rank": gt_rank,
                "num_valid_unique": len(solutions),
                "num_raw_completed": info.get("num_raw_completed", np.nan),
                "seconds": info.get("seconds", np.nan),
                "astar_expansions": info.get("expansions", np.nan),
                "astar_frontier_size": info.get("frontier_size", np.nan),
                "astar_stopped_by_budget": info.get(
                    "stopped_by_budget", False
                ),
                "error": info.get("error"),
            }
        )

search_df = pd.DataFrame(rows)
if search_df.empty:
    raise RuntimeError(
        "No comparison results were found. Inspect the worker output above."
    )

method_order = ["dfs", "astar_h0", "beam", "greedy", "top_p"]
k = records[0]["k"]
summary_rows = []

for method in method_order:
    d = search_df[search_df["method"] == method]
    if d.empty:
        continue

    summary_rows.append(
        {
            "method": method,
            "test_outputs": len(d),
            "puzzles": d["puzzle_id"].nunique(),
            "top1_exact": d["top1_exact"].mean(),
            f"recall@{k}": d["recall_at_k"].mean(),
            "MRR": np.mean(
                [1.0 / rank if pd.notna(rank) else 0.0 for rank in d["gt_rank"]]
            ),
            "mean_valid_unique": d["num_valid_unique"].mean(),
            "mean_raw_completed": d["num_raw_completed"].mean(),
            "median_seconds": d["seconds"].median(),
            "error_rate": d["error"].notna().mean(),
            "astar_mean_expansions": d["astar_expansions"].mean(),
            "astar_budget_stop_rate": (
                d["astar_stopped_by_budget"].mean()
                if method == "astar_h0"
                else np.nan
            ),
        }
    )

search_summary_df = pd.DataFrame(summary_rows)

display(search_summary_df)
display(
    search_df.sort_values(["puzzle_id", "test_output_id", "method"])
)

summary_path = "/kaggle/working/search_methods_summary.csv"
detail_path = "/kaggle/working/search_methods_per_item.csv"
search_summary_df.to_csv(summary_path, index=False)
search_df.to_csv(detail_path, index=False)

print("Saved:", summary_path)
print("Saved:", detail_path)
print(
    "Interpretation note: recall@4 is the main candidate-generation metric. "
    "Greedy can return only one candidate. A* is h=0 uniform-cost search; "
    "check its budget-stop rate before judging its recall."
)