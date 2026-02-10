# Guía de Inicio Rápido
## BezanillaSL: Sistema de Webscraping para Real Estate Analytics

Esta guía te permitirá empezar a usar el sistema en **menos de 10 minutos**.

---

## Instalación Express (5 minutos)

```bash
# 1. Navegar al proyecto
cd BezanillaSL/src/scraping

# 2. Instalar dependencias
pip install -r ../../scraping_requirements.txt

# 3. Instalar navegador
playwright install chromium

```

---

## Primer Scraping (2 minutos)

### Opción 1: Test Rápido (Solo Idealista, 2 páginas)

```bash
python main.py --portals idealista --pages 2 --format csv
```

**Resultado:** Archivo CSV en `data/scraped/processed/`

### Opción 2: Scraping Completo (Todos los portales)

```bash
python main.py --pages 15 --headless --format all
```

**Resultado:** CSV + Excel + JSON + Informe Resumen

### Opción 3: Portales Específicos

```bash
# Solo Idealista y Fotocasa
python main.py --portals idealista fotocasa --pages 10

# Solo Airbnb
python main.py --portals airbnb
```

---

## Ver los Datos

Los archivos se guardan automáticamente en:
```
data/scraped/processed/
├── cantabria_alquileres_TIMESTAMP.csv
├── cantabria_alquileres_TIMESTAMP.xlsx
├── cantabria_alquileres_TIMESTAMP.json
└── resumen_scraping_TIMESTAMP.xlsx  ← Abre este para ver estadísticas
```

---

## Comandos Útiles

### Ver opciones disponibles
```bash
python main.py --help
```

### Ejecutar con interfaz gráfica (ver el navegador)
```bash
python main.py --portals idealista --pages 2  # Sin --headless
```

### Exportar solo a Excel
```bash
python main.py --format excel
```

### Ver logs en tiempo real
```bash
tail -f ../../logs/scraping_$(date +%Y%m%d).log
```

---

## Configuración Rápida

Edita `config/settings.py` para cambiar:

```python
# Más páginas por defecto
MAX_PAGES_DEFAULT = 20

# Navegador invisible
HEADLESS_MODE = True

# Delays más lentos (si te bloquean)
DELAYS = {
    'page_load': (3000, 6000),
    'between_pages': (5000, 10000),
}
```

## Checklist de Verificación

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas (`pip install -r scraping_requirements.txt`)
- [ ] Playwright configurado (`playwright install chromium`)
- [ ] Test ejecutado exitosamente (`python main.py --pages 2`)
- [ ] Datos generados en `data/scraped/processed/`
- [ ] Configuración personalizada (opcional)

---

**¿Listo para empezar?** 

```bash
cd src/scraping && python main.py --pages 5
```

---

**Última actualización:** Febrero 2026  
**Versión:** 1.0  
**Tiempo estimado:** 10 minutos total