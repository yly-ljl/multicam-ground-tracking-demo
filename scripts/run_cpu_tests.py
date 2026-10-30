"""Run preserved CPU tests; explicitly omit two calibration tests unless requested."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CALIBRATION_TESTS = {
    'test_core.CoreTests.test_metric_projection_and_opencv',
    'test_improvement.WeightTests.test_analytic_jacobian_matches_finite_difference',
}


def leaves(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from leaves(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-calibration', action='store_true', help='Require local WILDTRACK C1/C2/C3 calibration files and include all tests')
    args = parser.parse_args()
    # Existing output-protection test requires the runtime root to exist.
    (ROOT / 'outputs').mkdir(exist_ok=True)
    tests = list(leaves(unittest.defaultTestLoader.discover(str(ROOT / 'tests'))))
    excluded = [] if args.with_calibration else sorted(CALIBRATION_TESTS)
    for name in excluded:
        print(f'NOT RUN (requires external calibration): {name}', flush=True)
    selected = unittest.TestSuite(t for t in tests if t.id() not in excluded)
    result = unittest.TextTestRunner(verbosity=2).run(selected)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
