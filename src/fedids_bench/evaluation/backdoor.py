"""
Backdoor attack success rate (ASR) of the FINAL global model.

The backdoor client stamps a trigger (the first ``trigger_dims`` standardized features set to ``trigger_value``) on a
share of its samples and relabels them ``target_label`` (default: benign).  A defense "works" only if triggered
attack flows are NOT waved through.  The F1 of the clean test set cannot show this, so it is measured directly:

    asr         share of true attack flows that the model classifies as ``target_label`` WITH the trigger,
    clean_miss  the same share WITHOUT the trigger (the ordinary false-negative rate, i.e. the floor),
    asr_lift    asr - clean_miss  (what the backdoor adds on top of ordinary misses).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from fedids_bench.config import AttackConfig


def apply_trigger(X: np.ndarray, trigger_value: float, trigger_dims: int) -> np.ndarray:
    Xt = np.array(X, copy=True)
    Xt[:, : min(int(trigger_dims), Xt.shape[1])] = trigger_value
    return Xt


def is_backdoor_run(attack: AttackConfig) -> bool:
    return attack.type == "backdoor" or (attack.type == "sample_inflation" and attack.combine_with == "backdoor")


def backdoor_success(model, X: np.ndarray, y: np.ndarray, attack: AttackConfig) -> Dict[str, Any]:
    """ASR on the evaluation split; ``X`` must be the standardized features the model was trained on."""
    target = int(attack.target_label) if attack.target_label is not None else 0
    victims = np.asarray(y) != target
    n = int(victims.sum())
    out: Dict[str, Any] = {
        "target_label": target, "n_victims": n, "trigger_value": float(attack.trigger_value),
        "trigger_dims": int(attack.trigger_dims), "asr": None, "clean_miss": None, "asr_lift": None,
    }
    if n == 0:
        return out
    Xv = np.asarray(X)[victims]
    clean_pred = np.argmax(model.predict_proba(Xv), axis=-1)
    trig_pred = np.argmax(model.predict_proba(apply_trigger(Xv, attack.trigger_value, attack.trigger_dims)), axis=-1)
    out["clean_miss"] = float((clean_pred == target).mean())
    out["asr"] = float((trig_pred == target).mean())
    out["asr_lift"] = out["asr"] - out["clean_miss"]
    return out
