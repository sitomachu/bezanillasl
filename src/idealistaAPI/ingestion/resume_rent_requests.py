from __future__ import annotations

import argparse

from src.idealistaAPI.ingestion.services.request_service import run_resume_latest_rent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resume latest RENT run from its manifest and continue until max requests."
    )
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--max-pages-per-circle", type=int, default=None)
    p.add_argument("--output-csv", type=str, default=None)
    p.add_argument("--no-adaptive-pages", action="store_true")
    p.add_argument(
        "--force-max-requests",
        action="store_true",
        help="Consume requests until max-requests even if pages are empty/partial.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = run_resume_latest_rent(
        max_requests_override=int(args.max_requests) if args.max_requests is not None else None,
        max_pages_per_circle_override=int(args.max_pages_per_circle) if args.max_pages_per_circle is not None else None,
        output_csv_override=str(args.output_csv) if args.output_csv else None,
        no_adaptive_pages=bool(args.no_adaptive_pages),
        force_max_requests=True if args.max_requests is not None else bool(args.force_max_requests),
    )
    print(f"OK. Results in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
