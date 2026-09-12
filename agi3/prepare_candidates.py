"""Build a seed-matched control and a candidate that reserves 90% of clicks for coverage."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_SEED = '        self.rng = random.Random(0xA3C ^ (hash(self.game_id) & 0xFFFF) ^ (seed_base * 0x9E37))'
NEW_SEED = '''        game_key = self.game_id.split("-")[0]
        game_seed = int.from_bytes(hashlib.sha256(game_key.encode()).digest()[:4], "big")
        self.rng = random.Random(0xA3C ^ game_seed ^ (seed_base * 0x9E37))'''
CLICK_POLICY = '''    def _next_click(self, latest_frame: FrameData) -> tuple[int, int] | None:
        """One object probe per ten clicks; all other clicks advance the sweep.

        No location is permanently excluded. In a stable level with only clicks,
        all 4096 pixels are visited within 4552 clicks, even when object probes
        keep choosing the same location. Level transitions restart the sweep.
        """
        self.click_calls += 1
        if self.click_calls % 10 == 1:
            targets = _components(_layer(latest_frame.frame))
            if targets:
                return self.rng.choice(targets)
        target = SWEEP[self.sweep_pos % len(SWEEP)]
        self.sweep_pos += 1
        return target

'''


def main():
    source = (HERE / "my_agent.py").read_text(encoding="utf-8")
    if source.count(OLD_SEED) != 1:
        raise ValueError("Baseline seed changed; inspect before rebuilding the control.")
    control = source.replace("import os\n", "import hashlib\nimport os\n", 1).replace(OLD_SEED, NEW_SEED)
    candidate = control.replace("        self.sweep_pos = 0\n", "        self.sweep_pos = 0\n        self.click_calls = 0\n", 1)
    start = candidate.index("    def _next_click(")
    end = candidate.index("    def _effect_rate(", start)
    candidate = candidate[:start] + CLICK_POLICY + candidate[end:]
    for name, body in (("stable_agent.py", control), ("coverage_agent.py", candidate)):
        compile(body, name, "exec")
        (HERE / name).write_text(body, encoding="utf-8")
        print(f"Built {name}")


if __name__ == "__main__":
    main()
