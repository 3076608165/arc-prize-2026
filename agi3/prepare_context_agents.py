"""Generate an isolated context-memory pair without editing the current trial.

Both standalone outputs have the same code except USE_CONTEXT_MEMORY. The
shared feedback-mask correction and sampling implementation belong to both
arms, so neither can be credited to the context-memory intervention.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXTENSION = '''
USE_CONTEXT_MEMORY = __ENABLED__


class MyAgent(CoverageBase):
    """Experimental context memory within the fixed 10% object-probe budget."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.context_memory = TransitionMemory()
        self.context = None
        self.context_probes = 0

    def _observe(self, latest_frame: FrameData) -> int:
        before_grid = self.prev_grid
        before_mask = self.noise_rows
        before_click = self.pending_click
        before_action = self.prev_action
        before_level = self.best_levels
        sig = super()._observe(latest_frame)
        mask_changed = before_mask != self.noise_rows
        level_changed = before_level != self.best_levels
        if mask_changed or level_changed:
            self.context_memory.clear()
        if (before_grid is not None and before_click is not None
                and before_action is not None and not mask_changed
                and not level_changed and self.steps_compared >= NOISE_WARMUP):
            self.context_memory.observe(self._signature(before_grid), before_click, sig)
        self.context = sig
        return sig

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.context_memory.clear()
            self.context = None
        return super().choose_action(frames, latest_frame)

    def _next_click(self, latest_frame: FrameData) -> tuple[int, int] | None:
        self.click_calls += 1
        if self.click_calls % 10 == 1:
            visible = _components(_layer(latest_frame.frame))
            target = self.context_memory.choose(
                self.context, visible, self.rng, enabled=USE_CONTEXT_MEMORY)
            if target is not None:
                self.context_probes += 1
                return target
        target = SWEEP[self.sweep_pos % len(SWEEP)]
        self.sweep_pos += 1
        return target
'''


def make_source(enabled):
    original = (HERE / "coverage_agent.py").read_text(encoding="utf-8")
    anchor = '        self._learn_noise(grid)\n        sig = self._signature(grid)\n'
    if original.count(anchor) != 1:
        raise ValueError("Coverage feedback changed; inspect before generating.")
    fixed = original.replace(anchor, '''        previous_grid = self.prev_grid
        previous_mask = self.noise_rows
        self._learn_noise(grid)
        sig = self._signature(grid)
        if previous_grid is not None:
            self.prev_sig = self._signature(previous_grid)
        if previous_mask != self.noise_rows:
            self.tried_simple.clear()
            self.effect.clear()
            self.responsive.clear()
            self.dead.clear()
''')
    tree = ast.parse(fixed)
    # Remove inherited commentary about withdrawn experiments from the new pair.
    tree.body[0] = ast.Expr(value=ast.Constant(
        "Generated context-memory prototype; unproven. Rebuild with "
        "prepare_context_agents.py. The pair differs only in USE_CONTEXT_MEMORY."))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MyAgent"]
    if len(classes) != 1:
        raise ValueError("Expected one source agent class")
    classes[0].name = "CoverageBase"
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n\n"
    memory = (HERE / "context_memory.py").read_text(encoding="utf-8")
    source += memory + "\n" + EXTENSION.replace("__ENABLED__", repr(enabled))
    compile(source, "context_agent.py", "exec")
    return source


def main():
    arms = {}
    for enabled, filename in ((False, "context_control_agent.py"), (True, "context_agent.py")):
        source = make_source(enabled)
        (HERE / filename).write_text(source, encoding="utf-8")
        arms[filename] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        print(f"Built {filename}")
    manifest = {"status": "prototype_not_promoted", "source_sha256": arms,
                "base_sha256": hashlib.sha256((HERE / "coverage_agent.py").read_bytes()).hexdigest(),
                "intervention": "Only USE_CONTEXT_MEMORY differs within the generated pair",
                "coverage_bound": "4096 pixels within 4552 click calls in a fixed level",
                "public_performance_claim": None}
    (HERE / "context_prototype.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
