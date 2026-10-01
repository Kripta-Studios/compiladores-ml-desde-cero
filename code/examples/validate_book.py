"""Execute the small worked numerical examples printed in the book."""
import argparse
import json
from pathlib import Path
import numpy as np
from lumbre import param, Program, gradients
from lumbre.nnops import conv2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('cpu', 'cuda', 'hip'), default='cpu')
    ap.add_argument('--output', type=Path, default=Path('reports/book_examples.json'))
    args = ap.parse_args()
    records = []
    x, w, b = param("book_X", (2, 2)), param("book_W", (2, 2)), param("book_b", (2,))
    y = x @ w + b
    loss = (y * y).mean()
    gs = gradients(loss, [x, w, b])
    with Program([y, loss, *gs], backend=args.backend, gemm="blocked") as p:
        out = p.run({"book_X": [[1, 2], [3, 4]], "book_W": np.eye(2), "book_b": [1, -1]})
    np.testing.assert_allclose(out[1], 7.5)
    np.testing.assert_allclose(out[2], [[1, .5], [2, 1.5]])
    np.testing.assert_allclose(out[3], [[7, 5], [10, 7]])
    np.testing.assert_allclose(out[4], [3, 2])
    records.append({"example": "linear_mse", "status": "PASS", "loss": float(out[1])})

    x, w = param("book_image", (1, 1, 3, 3)), param("book_filter", (1, 1, 2, 2))
    y = conv2d(x, w)
    gs = gradients(y.sum(), [x, w])
    with Program([y, *gs], backend=args.backend, gemm="blocked") as p:
        out = p.run({"book_image": np.arange(1, 10).reshape(1, 1, 3, 3),
                     "book_filter": np.array([1, 0, 0, -1]).reshape(1, 1, 2, 2)})
    np.testing.assert_allclose(out[0], -4)
    np.testing.assert_allclose(out[1][0, 0], [[1, 1, 0], [1, 0, -1], [0, -1, -1]])
    np.testing.assert_allclose(out[2][0, 0], [[12, 16], [24, 28]])
    records.append({"example": "convolution_gradients", "status": "PASS"})

    x, t = param("book_rx", (3,)), param("book_rt", (3,))
    w, b = param("book_rw", ()), param("book_rb", ())
    e = x * w + b - t
    loss = (e * e).mean()
    gw, gb = gradients(loss, [w, b])
    uw, ub = w - .1 * gw, b - .1 * gb
    with Program([loss, gw, gb, uw, ub], backend=args.backend) as p:
        old = p.run({"book_rx": [0, 1, 2], "book_rt": [1, 3, 5], "book_rw": 0, "book_rb": 0})
        np.testing.assert_allclose(old[1], -26 / 3, rtol=1e-6)
        np.testing.assert_allclose(old[2], -6, rtol=1e-6)
        p.assign({"book_rw": uw, "book_rb": ub})
        new = p.run(read=[loss])[0]
        assert new < old[0]
    records.append({"example": "regression_step", "status": "PASS", "before": float(old[0]), "after": float(new)})
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Three worked examples: PASS")


if __name__ == "__main__":
    main()
