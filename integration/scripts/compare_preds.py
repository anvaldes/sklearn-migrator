import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-in", required=True)
    ap.add_argument("--pred-out", required=True)
    args = ap.parse_args()

    a = np.load(args.pred_in)
    b = np.load(args.pred_out)

    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), "Predictions differ between environments"


if __name__ == "__main__":
    main()
