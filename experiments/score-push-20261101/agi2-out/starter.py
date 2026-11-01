import os
import time
import json
import traceback
import torch
import argparse
import torch.multiprocessing as mp

N_WORKERS = int(os.getenv("ARC_N_WORKERS", "4"))
FLAG_DIR = os.getenv("ARC_FLAG_DIR", "/kaggle")


def local_worker(rank, end_time, shared, test_path):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"{FLAG_DIR}/worker{rank-1}"):
            time.sleep(5)

    # A raised exception would make mp.spawn terminate the other three workers.
    worker = None
    try:
        from arc_solver import worker
    except BaseException as e:
        print(f"[Rank {rank}] FATAL import: {e!r}")
        traceback.print_exc()
    finally:
        with open(f"{FLAG_DIR}/worker{rank}", "w") as f:
            f.write("Ok")

    print(f"[Rank {rank}] start!")

    if worker is not None:
        try:
            worker(rank, end_time, shared, N_WORKERS, test_path)
        except BaseException as e:
            print(f"[Rank {rank}] FATAL: {e!r}")
            traceback.print_exc()

    print(f"[Rank {rank}] done!")


def resolve_test_path(rerun_mode):
    if os.getenv("ARC_TEST_PATH"):
        return os.getenv("ARC_TEST_PATH")
    name = "arc-agi_test_challenges.json" if rerun_mode else "arc-agi_evaluation_challenges.json"
    for root in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return os.path.join("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", name)


def main(worker_fn=local_worker):

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    from arc_sched import est_task_seconds, est_train_seconds, est_decode_seconds

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    test_path = resolve_test_path(rerun_mode)

    with open(test_path, "r") as f:
        data = json.load(f)

    if not rerun_mode:
        keys = os.getenv("ARC_COMMIT_TASKS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
        data = {k: data[k] for k in keys if k in data}

    est = {k: est_task_seconds(t) for k, t in data.items()}
    # cheapest (and most often solved) tasks first: whatever the time pressure on the
    # hidden set turns out to be, it can only squeeze the large low-yield tail
    order = sorted(data.keys(), key=lambda k: (est[k], k))

    budget = args.end_time - time.time()
    print(f"[starter] {len(order)} tasks, predicted baseline work {sum(est.values())/3600:.2f} GPU-h, "
          f"available {N_WORKERS * budget / 3600:.2f} GPU-h, test_path={test_path}")

    manager = mp.Manager()
    shared = dict(
        lock=manager.Lock(),
        order=manager.list(order),
        est=manager.dict(est),
        state=manager.dict(qi=0, tr_act=0.0, tr_exp=0.0, dec_act=0.0, dec_exp=0.0,
                           rem_tr0=float(sum(est_train_seconds(t) for t in data.values())),
                           rem_dec0=float(sum(est_decode_seconds(t) for t in data.values()))),
        busy=manager.dict(),
        unc=manager.dict(),
        tail_done=manager.dict(),
        failed=manager.dict(),
    )

    mp.spawn(worker_fn, args=(args.end_time, shared, test_path), nprocs=N_WORKERS, join=True)

    st = dict(shared["state"])
    print(f"[starter] finished: main queue reached {st['qi']}/{len(order)}, tail runs {len(shared['tail_done'])}, "
          f"kappa train {(1500 + st['tr_act']) / (1500 + st['tr_exp']):.3f} decode {(1500 + st['dec_act']) / (1500 + st['dec_exp']):.3f}")


if __name__ == "__main__":
    main()
