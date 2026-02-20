from __future__ import annotations

import argparse

from src.idealistaAPI.ingestion.services.request_service import add_common_args, run_new


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run RENT requests and export cleaned CSV.")
    add_common_args(p, default_csv="rent_homes_cantabria_bezana_like_raw.csv")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = run_new(
        operation="rent",
        max_requests=int(args.max_requests),
        max_pages_per_circle=max(1, int(args.max_pages_per_circle)),
        output_csv_name=str(args.output_csv),
        no_adaptive_pages=bool(args.no_adaptive_pages),
    )
    print(f"OK. Results in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
