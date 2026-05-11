"""Load MEME filler32k episodes as dspy.Example. Identical to simba/simba_karpathy."""
import json
import random
from pathlib import Path
from typing import List, Dict

import dspy


def load_episode(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def build_example(episode: Dict, domain: str) -> dspy.Example:
    before_qs = episode["before_questions"]["questions"]
    after_qs = episode["after_questions"]["questions"]

    ex = dspy.Example(
        episode_id=episode["episode_id"],
        domain=domain,
        sessions=episode["sessions"],
        before_pos=episode["before_questions"]["position_after_session"] + 1,
        after_pos=episode["after_questions"]["position_after_session"] + 1,
        before_questions=before_qs,
        after_questions=after_qs,
    ).with_inputs("episode_id", "domain", "sessions", "before_pos", "after_pos",
                  "before_questions", "after_questions")
    return ex


def load_trainset_testset(
    data_dir: Path,
    n_train_per_domain: int = 5,
    n_test_per_domain: int = 3,
    seed: int = 42,
) -> tuple[List[dspy.Example], List[dspy.Example]]:
    rng = random.Random(seed)

    trainset = []
    testset = []

    for domain, folder in [("pl", "filler32k_pl"), ("sw", "filler32k_sw")]:
        ep_dir = data_dir / folder
        files = sorted(ep_dir.glob("episode_*.json"))
        rng.shuffle(files)

        train_files = files[:n_train_per_domain]
        test_files = files[n_train_per_domain : n_train_per_domain + n_test_per_domain]

        for fp in train_files:
            trainset.append(build_example(load_episode(fp), domain))
        for fp in test_files:
            testset.append(build_example(load_episode(fp), domain))

    rng.shuffle(trainset)
    rng.shuffle(testset)
    return trainset, testset
