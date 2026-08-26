from __future__ import annotations

import os
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsFillSymbol,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)


PROJECT_ROOT = Path(os.environ["DISSERTATION_DATA_ROOT"]).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "GLOBAL_BB_PARTICIPATION_OUTPUT_DIR",
        PROJECT_ROOT
        / "final_data_and_analysis"
        / "Downstream_Analysis_Descriptive_E2SFCA_Longitudinal_20260820"
        / "05_Global_BB_Join_Count_ExactHalo20_40",
    )
).expanduser().resolve()
GPKG_PATH = OUTPUT_DIR / "annual_hp_la_bb_join_participation.gpkg"
QGIS_DIR = OUTPUT_DIR / "qgis"

CLASSES = (
    (0, "Not HP–LA", "#EEEEEE"),
    (1, "Isolated HP–LA", "#FDDDCB"),
    (2, "1–2 HP–LA neighbours", "#F4A582"),
    (3, "3–4 HP–LA neighbours", "#D6604D"),
    (4, "5+ HP–LA neighbours", "#B2182B"),
)


def fill_symbol(colour: str) -> QgsFillSymbol:
    return QgsFillSymbol.createSimple(
        {
            "color": colour,
            "outline_color": "#FFFFFF",
            "outline_style": "solid",
            "outline_width": "0.05",
            "outline_width_unit": "MM",
            "joinstyle": "bevel",
        }
    )


def main() -> None:
    QGIS_DIR.mkdir(parents=True, exist_ok=True)
    qgis_prefix = os.environ.get(
        "QGIS_PREFIX_PATH", "/Applications/QGIS-LTR.app/Contents/MacOS"
    )
    QgsApplication.setPrefixPath(qgis_prefix, True)
    app = QgsApplication([], False)
    app.initQgis()

    try:
        for year in (2001, 2011, 2021):
            layer_name = f"bb_participation_{year}"
            uri = f"{GPKG_PATH}|layername={layer_name}"
            layer = QgsVectorLayer(uri, layer_name, "ogr")
            if not layer.isValid():
                raise RuntimeError(f"Could not load {layer_name} from {GPKG_PATH}")

            categories = [
                QgsRendererCategory(code, fill_symbol(colour), label)
                for code, label, colour in CLASSES
            ]
            layer.setRenderer(QgsCategorizedSymbolRenderer("bb_class_code", categories))

            if year == 2001:
                message, ok = layer.saveNamedStyle(str(QGIS_DIR / "bb_participation_style.qml"))
                if not ok:
                    raise RuntimeError(message)
            error = layer.saveStyleToDatabase(
                "BB join participation",
                "Five-class annual HP-LA BB-neighbour participation style",
                True,
                "",
            )
            if error:
                raise RuntimeError(error)

        icb_uri = f"{GPKG_PATH}|layername=icb_boundaries_2023"
        icb = QgsVectorLayer(icb_uri, "icb_boundaries_2023", "ogr")
        if not icb.isValid():
            raise RuntimeError(f"Could not load ICB boundary layer from {GPKG_PATH}")
        icb.setRenderer(
            QgsSingleSymbolRenderer(
                QgsFillSymbol.createSimple(
                    {
                        "color": "0,0,0,0",
                        "outline_color": "#4D4D4D",
                        "outline_style": "solid",
                        "outline_width": "0.45",
                        "outline_width_unit": "MM",
                    }
                )
            )
        )
        message, ok = icb.saveNamedStyle(str(QGIS_DIR / "icb_boundaries_style.qml"))
        if not ok:
            raise RuntimeError(message)
        error = icb.saveStyleToDatabase(
            "ICB boundaries",
            "Transparent fill with dark-grey ICB outlines",
            True,
            "",
        )
        if error:
            raise RuntimeError(error)
    finally:
        app.exitQgis()

    print(f"Embedded default styles in: {GPKG_PATH}")
    print(f"Saved reusable QML: {QGIS_DIR / 'bb_participation_style.qml'}")
    print(f"Saved reusable QML: {QGIS_DIR / 'icb_boundaries_style.qml'}")


if __name__ == "__main__":
    main()
