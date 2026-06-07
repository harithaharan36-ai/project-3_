"""
Python client example for the CIFAR-10 API.

Usage
-----
    python examples/sample_request.py --image examples/cat.jpg
    python examples/sample_request.py --url   https://example.com/cat.jpg
"""

import argparse
import json
import sys

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--image", help="Local image file to upload")
    ap.add_argument("--url",   help="Remote image URL to classify")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    if not (args.image or args.url):
        ap.error("Provide either --image or --url")

    print(f"\n→ GET  {args.base}/health")
    print(json.dumps(requests.get(f"{args.base}/health").json(), indent=2))

    if args.image:
        print(f"\n→ POST {args.base}/predict?topk={args.topk}  (file={args.image})")
        with open(args.image, "rb") as fh:
            r = requests.post(
                f"{args.base}/predict",
                params={"topk": args.topk},
                files={"file": (args.image, fh, "image/jpeg")},
            )
    else:
        print(f"\n→ POST {args.base}/predict_url  (url={args.url})")
        r = requests.post(
            f"{args.base}/predict_url",
            json={"url": args.url, "topk": args.topk},
        )

    print(f"Status: {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
