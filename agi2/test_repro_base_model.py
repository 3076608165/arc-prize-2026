import json
import unittest
from pathlib import Path

from repro_nvarc import mount_path, set_base_model


def notebook(source: str) -> dict:
    return {"cells": [{"cell_type": "markdown", "source": ["x"]},
                      {"cell_type": "code", "source": [source]}]}


class MountPathTests(unittest.TestCase):
    def test_reproduces_the_upstream_mount(self):
        """The one ref/path pair we can verify against a running notebook.

        Upstream's metadata attaches the capitalised framework and its code
        reads the lowercased mount. Getting it backwards costs a full GPU run
        to find out, because the notebook dies on model load, minutes in.
        """
        self.assertEqual(
            mount_path("sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"),
            "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139"
            "/transformers/bfloat16/1")

    def test_lowercases_only_the_framework_segment(self):
        self.assertTrue(mount_path("owner/Big-Model/PyTorch/default/3")
                        .endswith("/owner/Big-Model/pytorch/default/3"))

    def test_rejects_a_ref_without_five_segments(self):
        for bad in ("owner/slug/Transformers/1", "slug", ""):
            with self.assertRaises(ValueError, msg=bad):
                mount_path(bad)


class SetBaseModelTests(unittest.TestCase):
    def test_rewrites_the_model_name_line(self):
        nb = notebook('model_name="/kaggle/input/models/old/small'
                      '/transformers/bfloat16/1",')
        self.assertTrue(set_base_model(nb, "new/big/Transformers/default/1"))
        self.assertEqual(
            nb["cells"][1]["source"][0],
            'model_name="/kaggle/input/models/new/big/transformers/default/1",')

    def test_reports_failure_instead_of_no_opping(self):
        """False becomes SystemExit in main().

        A silent no-op would push an unchanged notebook, and the resulting
        score would then be read as evidence about the new checkpoint.
        """
        self.assertFalse(set_base_model(notebook("model_path = 'x'"),
                                        "new/big/Transformers/default/1"))


class PreparedVariantTests(unittest.TestCase):
    def test_metadata_and_notebook_agree(self):
        """The notebook reads the mount path; metadata declares the ref.

        Both come from one argument in repro_nvarc.main(), so a mismatch means
        a variant was hand-edited -- and it would silently run whichever model
        the notebook still names, because an unattached source is absent.
        """
        for meta in sorted(Path(__file__).parent.glob("out/exp-*/kernel-metadata.json")):
            refs = json.loads(meta.read_text(encoding="utf-8")).get("model_sources") or []
            nb = Path(meta.parent, "submission.ipynb").read_text(encoding="utf-8")
            for ref in refs:
                self.assertIn(mount_path(ref), nb, f"{meta.parent.name}: {ref} unread")


if __name__ == "__main__":
    unittest.main()
