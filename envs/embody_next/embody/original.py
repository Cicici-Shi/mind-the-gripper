"""The 10 unperturbed libero_spatial tasks.

LIBERO-Plus's libero_spatial suite (2402 tasks) contains only perturbed variants;
every entry is listed in task_classification.json. The original bddl and
.pruned_init files are still on disk, so we rebuild Task objects for them.
"""
import os
import re

import torch

from libero.libero import get_libero_path
from libero.libero.benchmark import Task, grab_language_from_filename

# "_table_" alone would also drop the original task "..._from_table_center_...".
_PERTURBED = re.compile(r"_view_|_add_|_level|_light_|_table_\d|_tb_\d|_language_|_noise_| copy")


class OriginalSpatialSuite:
    folder = "libero_spatial"

    def __init__(self):
        init_dir = os.path.join(get_libero_path("init_states"), self.folder)
        names = sorted(
            f[: -len(".pruned_init")]
            for f in os.listdir(init_dir)
            if f.endswith(".pruned_init") and not _PERTURBED.search(f)
        )
        self.tasks = [
            Task(
                name=n,
                language=grab_language_from_filename(self.folder, n + ".bddl"),
                problem="Libero",
                problem_folder=self.folder,
                bddl_file=n + ".bddl",
                init_states_file=n + ".pruned_init",
            )
            for n in names
        ]
        self.n_tasks = len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        path = os.path.join(get_libero_path("init_states"), self.folder, self.tasks[i].init_states_file)
        return torch.load(path)


class OriginalObjectSuite(OriginalSpatialSuite):
    """The 10 original libero_object tasks (block 3, protocol_block3.md section 6), rebuilt the same way.
    Task order is the sorted file names, as for spatial; max steps 280 (set by the eval scripts)."""
    folder = "libero_object"


# ---------------- embody_next (2026-11-06): LIBERO goal / LIBERO-10 ----------------
# Same reconstruction as vla_zoo/g10/original_g10.py (src_goal10 baselines, sha256 47246089...a9aa9e): task order =
# sorted file names, NOT the official LIBERO order (goal ours->official [0, 3, 5, 8, 1, 4, 6, 9, 2, 7], libero_10
# [2, 3, 9, 8, 7, 0, 1, 4, 6, 5]); init = RandomState(5478 + task).permutation(50)[ep] in the eval scripts.
# Scene types (embody/scenes.py): goal = LIBERO_Tabletop_Manipulation for all 10 tasks; libero_10 (sorted) tasks 0-3
# Kitchen, 4-8 Living_Room, 9 Study. Additive only: the spatial / object classes above are unchanged.
class OriginalGoalSuite(OriginalSpatialSuite):
    folder = "libero_goal"


class Original10Suite(OriginalSpatialSuite):
    folder = "libero_10"


SUITES = {"libero_spatial": OriginalSpatialSuite, "libero_object": OriginalObjectSuite,
          "libero_goal": OriginalGoalSuite, "libero_10": Original10Suite}
# policy-step limits of the eval scripts (OFT run_libero_eval TASK_MAX_STEPS; openpi uses the same numbers)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
