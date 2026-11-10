"""STATE.json bookkeeping: stage, status, timestamps, SHA-256 hashes of key outputs."""
import os, json, hashlib, datetime, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FN = os.path.join(ROOT, "STATE.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update(stage, status, outputs=(), note=""):
    st = json.load(open(FN)) if os.path.exists(FN) else dict(stages={}, version_log=[])
    hashes = {}
    for pat in outputs:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            hashes[os.path.relpath(p, ROOT)] = sha(p)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    st["stages"][str(stage)] = dict(status=status, time=now, note=note, hashes=hashes)
    st["current_stage"] = str(stage); st["updated"] = now
    st["version_log"].append(dict(time=now, stage=str(stage), status=status, note=note))
    json.dump(st, open(FN, "w"), indent=1)


if __name__ == "__main__":
    import sys
    update(sys.argv[1], sys.argv[2], sys.argv[3:-1] if len(sys.argv) > 4 else sys.argv[3:], sys.argv[-1] if len(sys.argv) > 4 else "")
