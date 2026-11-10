"""Job queue for all PINN realizations.

Resumable (finished outputs are skipped) and safe for several parallel workers: a worker claims a
job by atomically creating results/pinn/<tag>.lock. Job order is rep-major so that, at any time, the
completed set is a balanced subset of the pre-registered matrix (DECISIONS.md D-20).
"""
import os, sys, json, time, traceback, socket
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import yaml  # noqa: E402
from train import run, tag_of, OUT  # noqa: E402

S = yaml.safe_load(open(os.path.join(ROOT, "configs", "sparse.yaml")))


def main_block(rep):
    J = []
    for rho in S["rho_levels"]:
        J.append(dict(scenario="B", rho=rho, eta=0.0, rep=rep, variant="pinn"))
    for eta in S["eta_levels"]:
        for rho in S["eta_rho_levels"]:
            J.append(dict(scenario="C", rho=rho, eta=eta, rep=rep, variant="pinn"))
    return J


def ablation_block(rep):
    J = [dict(scenario="B", rho=rho, eta=0.0, rep=rep, variant="dataonly") for rho in (0.20, 0.05)]
    J.append(dict(scenario="C", rho=0.05, eta=0.10, rep=rep, variant="dataonly"))
    if rep < 5:   # ablation: fixed unit weights instead of adaptive weights
        J.append(dict(scenario="B", rho=0.05, eta=0.0, rep=rep, variant="noadapt"))
    return J


def jobs():
    A = [dict(scenario="A", rho=1.0, eta=0.0, rep=0, variant="pinn", init=i) for i in range(S["n_init_A"])]
    J = [A[0]]
    for rep in range(5):
        J += main_block(rep)
    J += A[1:]
    for rep in range(5):
        J += ablation_block(rep)
    for rep in range(5, S["n_masks"]):
        J += main_block(rep)
    for rep in range(5, S["n_masks"]):
        J += ablation_block(rep)
    return J


def claim(tag):
    if os.path.exists(os.path.join(OUT, tag + ".npz")):
        return False
    try:
        fd = os.open(os.path.join(OUT, tag + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


if __name__ == "__main__":
    J = jobs()
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(ROOT, "results", "logs", "queue_progress.jsonl")
    for k, j in enumerate(J):
        tag = tag_of(j["scenario"], j["rho"], j["eta"], j["rep"], j["variant"], init=j.get("init"))
        if not claim(tag):
            continue
        t0 = time.time()
        try:
            fn = run(j["scenario"], j["rho"], j["eta"], j["rep"], j["variant"], init=j.get("init"))
            st = "ok"
        except Exception:  # failures are logged, never silently dropped
            fn, st = traceback.format_exc(), "error"
        with open(log, "a") as f:
            f.write(json.dumps(dict(k=k, n=len(J), job=j, status=st, out=fn, wall=time.time() - t0,
                                    pid=os.getpid())) + "\n")
        if st == "ok":
            os.remove(os.path.join(OUT, tag + ".lock"))
    print("queue done")
