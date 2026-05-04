"""Sanity minimo del DataLoader de OpenGait sobre nuestros pkl.

Objetivo: confirmar que opengait/data/dataset.py:DataSet parsea correctamente
data/pkl/<sujeto>/<cond>/090/seq00.pkl y devuelve numpy (T,64,44) uint8 antes
de meter el modelo.

No entrena nada. No carga checkpoint. Solo:
1. inicializa torch.distributed con backend gloo (1 proc) — requerido por get_msg_mgr.
2. instancia DataSet con partition_zeroshot.json (TEST_SET=19 sujetos).
3. itera 3 muestras y verifica shape/dtype.
4. reporta tamano de la primera secuencia.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait" / "opengait"
sys.path.insert(0, str(OPENGAIT))


def init_single_proc_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # Win + PyTorch sin libuv: hay que desactivar antes de TCPStore.
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def main() -> int:
    init_single_proc_dist()

    from data.dataset import DataSet  # type: ignore  # noqa: E402
    from utils import get_msg_mgr  # type: ignore  # noqa: E402

    tmp_log_dir = REPO / "reports" / "_validator_logs"
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    get_msg_mgr().init_manager(str(tmp_log_dir), log_to_file=False, log_iter=100)

    data_cfg = {
        "dataset_root": str(REPO / "data" / "pkl"),
        "dataset_partition": str(REPO / "configs" / "partition_zeroshot.json"),
        "cache": False,
        "num_workers": 0,
    }

    ds = DataSet(data_cfg, training=False)
    n = len(ds)
    print(f"[ok] DataSet construido. n_secuencias={n}")
    print(f"[ok] sujetos detectados ({len(ds.label_set)}): {ds.label_set[:3]} ... {ds.label_set[-3:]}")
    print(f"[ok] tipos detectados: {ds.types_set}")
    print(f"[ok] vistas detectadas: {ds.views_set}")

    if n == 0:
        print("[err] DataSet vacio — partition o dataset_root mal configurados")
        return 1

    expected = 19 * 2  # 19 sujetos x 2 condiciones x 1 vista x 1 seq
    if n != expected:
        print(f"[warn] esperaba {expected} secuencias, encontre {n}")

    failures = 0
    for i in (0, n // 2, n - 1):
        data_list, seq_info = ds[i]
        arr = data_list[0]
        label, typ, view = seq_info[:3]
        if not isinstance(arr, np.ndarray):
            print(f"[err] [{i}] tipo inesperado: {type(arr)}")
            failures += 1
            continue
        ok_shape = arr.ndim == 3 and arr.shape[1:] == (64, 44)
        ok_dtype = arr.dtype == np.uint8
        status = "ok" if (ok_shape and ok_dtype) else "err"
        print(f"[{status}] [{i:>2}] {label}/{typ}/{view}  shape={arr.shape}  dtype={arr.dtype}")
        if not (ok_shape and ok_dtype):
            failures += 1

    if failures == 0:
        print("\n[done] DataLoader OpenGait compatible con nuestros pkl. Listo para zero-shot.")
        return 0
    print(f"\n[done] {failures} fallos — revisar antes de continuar")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
