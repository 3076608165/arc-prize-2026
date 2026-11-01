"""Cost model and budget pacing for the 4-worker TTT pipeline.

The cost model was fitted on 24 ARC-AGI-2 evaluation tasks run with the baseline
configuration (128 TTT sequences, 16 views, DFS p>=0.2) on one L4:
  train  ~ 19.2 + 0.0809*S + 4.16e-6*S^2   seconds per 128 steps (S = training sequence tokens)
  decode ~ 185 + 0.588*OT - 0.0381*OT*S/1000   seconds for 16 views (OT = predicted output tokens)
Training is predicted within ~10 s, decoding only within ~230 s, so the pacer calibrates
a global factor kappa = actual / predicted online and re-plans every task from the
remaining wall time.
"""
import time

import numpy as np

TRAIN_COEF = (19.2334, 0.0809330, 4.16404e-6)
DECODE_COEF = (185.248, 0.588264, -0.0381164)
SETUP_S = 15.0

# Level 0 = baseline config. The baseline stops starting decode batches 1200 s into a task
# and lets the running batch finish (up to its DFS window), so tasks can last ~1700 s; the
# cap reproduces that. Boost levels are disabled: the v8 run with them scored 26.39.
LEVELS = {
    0: dict(n_train_aug=16, extra_views=False, factor=1.00, cap=1700.0),
    1: dict(n_train_aug=16, extra_views=True,  factor=1.30, cap=1700.0),
    2: dict(n_train_aug=24, extra_views=True,  factor=1.55, cap=1900.0),
}
LEVEL1_RHO = float("inf")
LEVEL2_RHO = float("inf")
KAPPA_PRIOR_WEIGHT = 1500.0

PASS_SEEDS = {
    "A":  dict(train_seed=1, lora_seed=42,  eval_seed=2, suffix=""),
    "B":  dict(train_seed=17, lora_seed=137, eval_seed=29, suffix=".p2"),
}


def seq_tokens(task):
    return sum(len(g) * (len(g[0]) + 1) + 6 for p in task["train"] for g in (p["input"], p["output"]))


def predicted_output_tokens(task, test_item):
    ins = [np.shape(p["input"]) for p in task["train"]]
    outs = [np.shape(p["output"]) for p in task["train"]]
    x = np.shape(test_item["input"])
    if all(a == b for a, b in zip(ins, outs)):
        o = x
    elif len(set(outs)) == 1:
        o = outs[0]
    else:
        ry = np.mean([b[0] / a[0] for a, b in zip(ins, outs)])
        rx = np.mean([b[1] / a[1] for a, b in zip(ins, outs)])
        o = (max(1, min(30, round(x[0] * ry))), max(1, min(30, round(x[1] * rx))))
    return o[0] * (o[1] + 1) + 1


def est_train_seconds(task, n_train_aug=16):
    S = min(seq_tokens(task), 8192)
    per128 = TRAIN_COEF[0] + TRAIN_COEF[1] * S + TRAIN_COEF[2] * S * S
    return per128 * n_train_aug / 16


def est_decode_seconds(task):
    S = min(seq_tokens(task), 8192)
    total = 0.0
    for t in task["test"]:
        ot = predicted_output_tokens(task, t)
        total += DECODE_COEF[0] + DECODE_COEF[1] * ot + DECODE_COEF[2] * ot * S / 1000
    # the fit is per task (one test output for most tasks); extra outputs add less than a full task
    n = len(task["test"])
    total = total * (1 + 0.6 * (n - 1)) / n
    return float(min(max(total, 60.0), 1300.0))


def est_task_seconds(task):
    return SETUP_S + est_train_seconds(task) + est_decode_seconds(task)


def output_uncertainty(u):
    """0 = settled, higher = worth another independent TTT run."""
    if u["c1"] == 0:
        return 0.25  # no candidate at all: a second run rarely recovers these
    if u["c1"] >= max(6, u["views"] // 2) and u["c1"] >= 2 * u["c2"] + 2:
        return 0.0
    return 1.0 - u["c1"] / max(u["views"], 1) + 0.25 * min(u["n_unique"], 4) / 4


class Pacer:
    """Hands out tasks and per-task wall budgets to one worker.

    shared: dict of Manager proxies created by starter.py
      lock, order (list of keys, cheap first), est (key -> base seconds),
    tasks: key -> ARC task (train/test inputs), for the training-time floor
      state (qi, rem, act, exp), busy (rank -> busy-until ts),
      unc (key -> per-output uncertainty), tail_done (key -> 1), failed (key -> 1)
    """

    def __init__(self, shared, rank, n_workers, end_time, tasks):
        self.s = shared
        self.tasks = tasks
        self.rank = rank
        self.n_workers = n_workers
        self.end_time = end_time

    def kappas(self):
        st = self.s["state"]
        kt = (KAPPA_PRIOR_WEIGHT + st["tr_act"]) / (KAPPA_PRIOR_WEIGHT + st["tr_exp"])
        kd = (KAPPA_PRIOR_WEIGHT + st["dec_act"]) / (KAPPA_PRIOR_WEIGHT + st["dec_exp"])
        return kt, kd

    def kappa(self):
        """Overall actual/predicted ratio for a typical task mix."""
        kt, kd = self.kappas()
        st = self.s["state"]
        return (kt * st["rem_tr0"] + kd * st["rem_dec0"]) / max(st["rem_tr0"] + st["rem_dec0"], 1.0)

    def task_seconds(self, key, level=0):
        kt, kd = self.kappas()
        lv = LEVELS[level]
        t = self.tasks[key]
        dec_factor = 1.35 if lv["extra_views"] else 1.0
        return SETUP_S + kt * est_train_seconds(t, lv["n_train_aug"]) + kd * est_decode_seconds(t) * dec_factor

    def available(self, now):
        others = 0.0
        for r, until in self.s["busy"].items():
            if r != self.rank:
                others += max(0.0, until - now)
        return self.n_workers * max(0.0, self.end_time - now) - others

    def _plan(self, key, pass_name, rem_tr, rem_dec, now):
        kt, kd = self.kappas()
        A = self.available(now)
        task = self.tasks[key]
        base = self.task_seconds(key, 0)
        D = kt * rem_tr + kd * rem_dec + base
        rho = A / max(D, 1.0)
        if pass_name == "A":
            level = 2 if rho >= LEVEL2_RHO else (1 if rho >= LEVEL1_RHO else 0)
        else:
            level = 0
        lv = LEVELS[level]
        expected = self.task_seconds(key, level)
        train_floor = SETUP_S + kt * est_train_seconds(task, lv["n_train_aug"]) + 90.0
        if rho >= 1.0:
            # on schedule: give the task the full baseline allowance, never less
            budget = lv["cap"]
        else:
            # behind schedule: shrink this task's share in proportion
            budget = min(lv["cap"], 1.2 * base * max(rho, 0.25))
        budget = max(budget, train_floor)
        budget = min(budget, self.end_time - now - 20.0)
        plan = dict(task=key, pass_name=pass_name, level=level, est=self.s["est"][key], kappa=kd, rho=rho,
                    budget=budget, expected=min(budget, expected),
                    est_train=est_train_seconds(task, lv["n_train_aug"]), est_dec=est_decode_seconds(task) * (1.35 if lv["extra_views"] else 1.0),
                    n_train_aug=lv["n_train_aug"], extra_views=lv["extra_views"],
                    **PASS_SEEDS[pass_name])
        return plan, train_floor

    def next_task(self):
        while True:
            now = time.time()
            if now > self.end_time - 120:
                return None
            with self.s["lock"]:
                st = self.s["state"]
                qi = st["qi"]
                order = self.s["order"]
                if qi < len(order):
                    key = order[qi]
                    task = self.tasks[key]
                    st["qi"] = qi + 1
                    st["rem_tr0"] = st["rem_tr0"] - est_train_seconds(task)
                    st["rem_dec0"] = st["rem_dec0"] - est_decode_seconds(task)
                    plan, train_floor = self._plan(key, "A", st["rem_tr0"], st["rem_dec0"], now)
                    if now + train_floor > self.end_time - 20:
                        print(f"[Rank {self.rank}] skip {key}: not enough time to train "
                              f"(needs {train_floor:.0f}s, {self.end_time - now:.0f}s left)")
                        continue
                    self.s["busy"][self.rank] = now + plan["expected"]
                    return plan
                # main queue exhausted -> tail pass on the most uncertain finished tasks
                plan = self._pick_tail(now)
                if plan is None:
                    self.s["busy"][self.rank] = now
                    return None
                self.s["busy"][self.rank] = now + plan["expected"]
                return plan

    def _pick_tail(self, now):
        left = self.end_time - now - 30.0
        best, best_score = None, 0.0
        done = self.s["tail_done"]
        failed = dict(self.s["failed"])
        for key, unc in list(self.s["unc"].items()) + [(k, None) for k in failed]:
            if key in done:
                continue
            est = self.task_seconds(key, 0)
            if est * 1.1 > left:
                continue
            if unc is None:
                score = 2.0
            else:
                us = [output_uncertainty(u) for u in unc.values()]
                score = float(np.mean(us)) if us else 0.0
            if score <= 0.05:
                continue
            score = score / (est / 600.0 + 0.5)
            if score > best_score:
                best, best_score = key, score
        if best is None:
            return None
        done[best] = 1
        est = self.task_seconds(best, 0)
        pass_name = "A" if best in failed else "B"
        plan, _ = self._plan(best, pass_name, 0.0, 0.0, now)
        plan["budget"] = min(max(plan["budget"], est), left)
        plan["expected"] = min(plan["budget"], est)
        return plan

    def done(self, plan, actual, info):
        with self.s["lock"]:
            st = self.s["state"]
            if info is not None and "train_s" in info:
                st["tr_act"] = st["tr_act"] + max(info["train_s"] - SETUP_S, 1.0)
                st["tr_exp"] = st["tr_exp"] + plan["est_train"]
                dec = max(actual - info["train_s"], 1.0)
                if info.get("timed_out", False):
                    dec *= 1.3  # censored: the decode would have run longer
                st["dec_act"] = st["dec_act"] + dec
                st["dec_exp"] = st["dec_exp"] + plan["est_dec"]
            self.s["busy"][self.rank] = time.time()
            if info is None:
                if plan["pass_name"] == "A":
                    self.s["failed"][plan["task"]] = 1
            elif plan["pass_name"] == "A":
                self.s["unc"][plan["task"]] = info["unc"]
                if plan["task"] in self.s["failed"]:
                    del self.s["failed"][plan["task"]]
