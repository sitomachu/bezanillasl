from __future__ import annotations

import argparse
from datetime import datetime

from src.idealistaAPI.config.idealista import RAW_BASE
from src.idealistaAPI.ingestion.services.request_service import add_common_args, run_new, run_resume_latest_rent
from src.idealistaAPI.processing.clean_idealista import clean_json_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run RENT requests and store raw responses.")
    add_common_args(p, default_csv="rent_homes_cantabria_bezana_like_raw.csv")
    p.add_argument(
        "--resume-latest",
        action="store_true",
        help="Continue the latest rent run before opening new searches in additional circles.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.resume_latest:
        processed_dir = run_resume_latest_rent(
            max_requests_override=int(args.max_requests),
            max_pages_per_circle_override=max(1, int(args.max_pages_per_circle)),
            output_csv_override=str(args.output_csv),
            no_adaptive_pages=bool(args.no_adaptive_pages),
            force_max_requests=bool(args.force_max_requests),
        )
    else:
        processed_dir = run_new(
            operation="rent",
            max_requests=int(args.max_requests),
            max_pages_per_circle=max(1, int(args.max_pages_per_circle)),
            output_csv_name=str(args.output_csv),
            no_adaptive_pages=bool(args.no_adaptive_pages),
            force_max_requests=bool(args.force_max_requests),
        )
    raw_dir = RAW_BASE / processed_dir.name
    csv_path = clean_json_run(input_dir=raw_dir, output_filename=str(args.output_csv))
    now = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{now}] Ejecucion finalizada. resultados_en={processed_dir.resolve()} csv_generado={csv_path.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
