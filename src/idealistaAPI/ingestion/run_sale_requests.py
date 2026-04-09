from __future__ import annotations

import argparse
from datetime import datetime

from src.idealistaAPI.ingestion.services.request_service import add_common_args, run_new


DEFAULT_SALE_CSV = "sale_homes_cantabria_bezana_like_raw.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SALE requests and store raw responses.")
    add_common_args(parser, default_csv=DEFAULT_SALE_CSV)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    processed_dir = run_new(
        operation="sale",
        max_requests=int(args.max_requests),
        max_pages_per_circle=max(1, int(args.max_pages_per_circle)),
        output_csv_name=str(args.output_csv),
        no_adaptive_pages=bool(args.no_adaptive_pages),
        force_max_requests=bool(args.force_max_requests),
    )
    now = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{now}] Ejecucion finalizada. resultados_en={processed_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
