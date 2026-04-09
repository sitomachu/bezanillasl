from __future__ import annotations

import argparse
from datetime import datetime

from src.idealistaAPI.ingestion.services.request_service import Location, add_common_args, run_new

# Segunda tanda de municipios costeros de Cantabria.
# Los de la primera tanda (Villaescusa, ElAstillero, MedioCudeyo, Polanco,
# SantaMariadeCayon, Torrelavega, Cartes, Santona, Noja, Entrambasaguas)
# ya fueron cubiertos en el run 20260401_144949.
#
# Los location_id siguen el formato INE: 0-EU-ES-39-{código_municipio}.
# Si la API devuelve 404 para alguno, el sistema cae automáticamente a center+distance.
EXTENDED_LOCATIONS: list[Location] = [
    # --- Costa oriental (entre zonas ya cubiertas) ---
    Location("Bareyo",               "0-EU-ES-39-39010", "43.4750,-3.6700", 5000),
    Location("Arnuero",              "0-EU-ES-39-39005", "43.4780,-3.5090", 4000),
    Location("BarcenadeCicero",      "0-EU-ES-39-39008", "43.4200,-3.4280", 5000),
    Location("RibamontanAlMonte",    "0-EU-ES-39-39057", "43.4000,-3.7600", 5000),
    # --- Costa occidental (no cubierta) ---
    Location("SantillanadelMar",     "0-EU-ES-39-39072", "43.3890,-4.1020", 5000),
    Location("AlfozDeLloredo",       "0-EU-ES-39-39001", "43.3750,-4.2080", 6000),
    Location("Comillas",             "0-EU-ES-39-39020", "43.3870,-4.2890", 5000),
    Location("SanVicenteBarquera",   "0-EU-ES-39-39068", "43.3880,-4.3990", 5000),
    # --- Suburbano interior (cerca de costa) ---
    Location("Reocin",               "0-EU-ES-39-39059", "43.3680,-4.0870", 5000),
    Location("LosCorralesDeBuelna",  "0-EU-ES-39-39026", "43.2620,-4.0680", 5000),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RENT requests sobre municipios extendidos (no incluidos en el run estándar)."
    )
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
        force_max_requests=bool(args.force_max_requests),
        locations=EXTENDED_LOCATIONS,
    )
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Ejecucion finalizada. resultados_en={out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
