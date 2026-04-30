from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rca.run_agent_standard import main


DATASET = "Telecom"
ROW_ID = 0

FAST_EASY_SOLVER = True
USE_ARCHIVE_BASELINE = False
ARCHIVE_PRED = ""
WINDOW_GRANGER_SOLVER = False
HYBRID_WINDOW_GRANGER_PRIOR = False
LIGHT_AGENT = False
LIGHT_AGENT_TEMPERATURE = 0.0
FORCE_COUNTERFACTUAL = False
NO_LLM_FALLBACK = False


def run() -> None:
    dataset = str(DATASET)
    rid = int(ROW_ID)
    q = pd.read_csv(f"dataset/{dataset}/query.csv")
    row = q.iloc[rid]
    task_index = str(row["task_index"])
    args = SimpleNamespace(
        dataset=dataset,
        sample_num=1,
        start_idx=rid,
        end_idx=rid,
        controller_max_step=25,
        controller_max_turn=5,
        timeout=600,
        tag="debug_session",
        auto=False,
        row_ids_file="",
        route_mode="auto",
        fast_easy_solver=bool(FAST_EASY_SOLVER),
        use_archive_baseline=bool(USE_ARCHIVE_BASELINE),
        archive_pred=str(ARCHIVE_PRED or ""),
        window_granger_solver=bool(WINDOW_GRANGER_SOLVER),
        hybrid_window_granger_prior=bool(HYBRID_WINDOW_GRANGER_PRIOR),
        light_agent=bool(LIGHT_AGENT),
        light_agent_temperature=float(LIGHT_AGENT_TEMPERATURE),
        force_counterfactual=bool(FORCE_COUNTERFACTUAL),
        no_llm_fallback=bool(NO_LLM_FALLBACK),
    )

    logger.info(
        json.dumps(
            {
                "dataset": dataset,
                "row_id": rid,
                "task_index": task_index,
            },
            ensure_ascii=False,
        )
    )
    uid = "debug_session"
    main(args, uid=uid, dataset=dataset)


if __name__ == "__main__":
    run()
