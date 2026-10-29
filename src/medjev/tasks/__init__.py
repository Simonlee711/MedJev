"""Task builders: dataset rows -> Jev requests (`Example`s) with gold labels.

BUILDERS:       evaluation sets (test / held-out splits)
TRAIN_BUILDERS: labelled training sets for recalibration and the supervised baselines (patient-disjoint from the
                evaluation sets)

Every builder takes an optional `limit`; by default the full split is used.
"""

from medjev.tasks import ed_triage, medqa, meds, qa

BUILDERS = {
    "medqa": medqa.build,
    "usmle4": qa.build_usmle4,
    "diagnosisarena": qa.build_diagnosisarena,
    "nejm_medqa": qa.build_nejm,
    **{name: (lambda split, limit, _n=name: meds.build(_n, split, limit)) for name in meds.TASKS},
    "ed_triage_admission": ed_triage.build_admission,
    "ed_triage_disposition": ed_triage.build_disposition,
}

TRAIN_BUILDERS = {
    "medqa": lambda limit: medqa.build_train(limit),
    **{name: (lambda limit, _n=name: meds.build(_n, "train", limit)) for name in meds.TASKS},
    "ed_triage_admission": lambda limit: ed_triage.build_admission("train", limit),
    "ed_triage_disposition": lambda limit: ed_triage.build_disposition("train", limit),
}

__all__ = ["BUILDERS", "TRAIN_BUILDERS"]
