"""embody_next (2026-11-06): which robot class LIBERO builds for a body in each scene type.

LIBERO's problem classes rewrite robots=[name] into f"{prefix}{name}" and place the base with
robot_model.base_xpos_offset[key] (libero/envs/bddl_base_domain.py _load_model). Only two prefixes exist:

  problem (bddl "(problem ...)")              prefix        base offset key      suites
  LIBERO_Tabletop_Manipulation                Mounted       table                spatial, goal
  LIBERO_Kitchen_Tabletop_Manipulation        Mounted       kitchen_table        libero_10 (sorted tasks 0-3)
  LIBERO_Study_Tabletop_Manipulation          Mounted       study_table          libero_10 (task 9)
  LIBERO_Living_Room_Tabletop_Manipulation    OnTheGround   living_room_table    libero_10 (tasks 4-8)
  LIBERO_Floor_Manipulation                   OnTheGround   empty                object
  LIBERO_Coffee_Table_Manipulation            OnTheGround   coffee_table         (none of ours)

embody/robots.py already registers Mounted<body> for every body (Panda base offsets, which include kitchen_table and
study_table) and OnTheGround<body> (OnTheGroundPanda offsets, which include living_room_table and coffee_table), so no
new class is needed for goal / 10; body_scene_class() makes that explicit and fails loudly if a body / scene pair
would not resolve. Read-only helper: importing it changes nothing.
"""
import re

PROBLEMS = {
    "libero_tabletop_manipulation": ("Mounted", "table"),
    "libero_kitchen_tabletop_manipulation": ("Mounted", "kitchen_table"),
    "libero_study_tabletop_manipulation": ("Mounted", "study_table"),
    "libero_living_room_tabletop_manipulation": ("OnTheGround", "living_room_table"),
    "libero_floor_manipulation": ("OnTheGround", "empty"),
    "libero_coffee_table_manipulation": ("OnTheGround", "coffee_table"),
}


def problem_name(bddl_file):
    with open(bddl_file) as f:
        m = re.search(r"\(problem\s+([A-Za-z_]+)\)", f.read())
    assert m, f"no (problem ...) in {bddl_file}"
    return m.group(1).lower()


def body_scene_class(body, bddl_file):
    """(class name LIBERO will build, base offset key, problem) for robots=[body] in this bddl; asserts it resolves."""
    from . import robots  # noqa: F401  registers the Mounted* / OnTheGround* classes
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING
    prob = problem_name(bddl_file)
    assert prob in PROBLEMS, f"unknown scene type {prob}"
    prefix, key = PROBLEMS[prob]
    cname = prefix + body
    assert cname in REGISTERED_ROBOTS and cname in ROBOT_CLASS_MAPPING, f"{cname} not registered ({prob})"
    offs = REGISTERED_ROBOTS[cname].base_xpos_offset.fget(None)
    assert key in offs, f"{cname}.base_xpos_offset has no {key!r}"
    return cname, key, prob
