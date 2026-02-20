from pathlib import Path
import json

from src.idealistaAPI.ingestion.client import IdealistaClient

def main():
    client = IdealistaClient()

    # 1 búsqueda = 1 request (POST /search)
    # Zona: Santander aprox, radio 15 km. Ajusta si quieres.
    resp = client.search(
        country="es",
        operation="rent",
        property_type="homes",
        num_page=1,
        max_items=50,
        center="43.4623,-3.8099",   # Santander aprox
        distance=15000,
        extra_params={
            "order": "publicationDate",
            "sort": "desc"
        }
    )


    # Exporta resultados
    out_dir = Path("data/raw/idealista/test")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "response_page1.json").write_text(
        json.dumps(resp, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Extra: guarda solo los anuncios (elementList) en JSONL
    element_list = resp.get("elementList", []) or []
    with (out_dir / "elementList.jsonl").open("w", encoding="utf-8") as f:
        for item in element_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"OK. Guardado en: {out_dir.resolve()}")
    print(f"Anuncios devueltos: {len(element_list)}")

if __name__ == "__main__":
    main()
