"""Run the Streamlit app headless and check it renders without errors (needs streamlit)."""

import os
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # streamlit not installed: skip
    AppTest = None

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py")


@unittest.skipIf(AppTest is None, "streamlit not installed")
class AppTests(unittest.TestCase):
    def test_app_runs(self):
        at = AppTest.from_file(APP, default_timeout=300).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertIn("Card fraud detection", at.title[0].value)
        self.assertGreaterEqual(len(at.metric), 5)

    def test_examples_score(self):
        at = AppTest.from_file(APP, default_timeout=300).run()
        for example in ("Card testing", "Stolen card abroad", "Account takeover at night"):
            radios = [r for r in at.radio if r.label == "Example"]
            radios[0].set_value(example).run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
