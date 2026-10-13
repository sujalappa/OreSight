"""Diagnostic absorption features measured from EMIT L2A reflectance.

Each feature is measured the same way (see ``band_depth.py``): a straight-line
continuum between two shoulders, continuum removal, and a parabolic fit to the
three bands around the minimum inside the search window. That gives a band depth
and the wavelength of the absorption minimum.

Shoulders avoid the atmospheric water bands (~940, 1140, 1380, 1880 nm).

These features are NOT unique to one mineral. The ``minerals`` text lists the
common causes; the position of the minimum (e.g. Al-OH 2195 vs 2215 nm) and the
EMIT L2B identifications are what narrow it down.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AbsorptionFeature:
    key: str
    label: str
    left_nm: float
    right_nm: float
    window_nm: tuple[float, float]
    minerals: str
    position_range_nm: tuple[float, float]  # display range for the position map


FEATURES: tuple[AbsorptionFeature, ...] = (
    AbsorptionFeature(
        "ferric_900", "Fe3+ crystal-field absorption (~900 nm)", 760.0, 1255.0, (850.0, 965.0),
        "hematite (~860-880 nm), goethite (~900-940 nm), jarosite; also ferruginous/lateritic soils",
        (850.0, 960.0)),
    AbsorptionFeature(
        "ferrous_1000", "Fe2+ crystal-field absorption (~1000 nm)", 760.0, 1255.0, (965.0, 1120.0),
        "Fe2+ in amphibole, pyroxene, olivine, chlorite, biotite (mafic / ultramafic rocks, basalt soils)",
        (970.0, 1100.0)),
    AbsorptionFeature(
        "aloh_2200", "Al-OH absorption (~2200 nm)", 2120.0, 2250.0, (2180.0, 2228.0),
        "white mica (muscovite/sericite/illite), kaolinite, smectite (montmorillonite), alunite; "
        "short wavelength (~2195 nm) = Na/Al-rich mica, long (~2215 nm) = phengitic mica",
        (2190.0, 2220.0)),
    AbsorptionFeature(
        "feoh_2250", "Fe-OH absorption (~2250 nm)", 2228.0, 2280.0, (2240.0, 2265.0),
        "chlorite, epidote, biotite, jarosite",
        (2240.0, 2265.0)),
    AbsorptionFeature(
        "mgoh_2330", "Mg-OH / carbonate absorption (~2330 nm)", 2270.0, 2385.0, (2300.0, 2360.0),
        "chlorite, epidote, amphibole, talc, serpentine (Mg-OH ~2310-2350 nm); "
        "calcite (~2340 nm), dolomite (~2320 nm)",
        (2305.0, 2355.0)),
)

FEATURE_BY_KEY = {f.key: f for f in FEATURES}

# Band windows for the ancillary indices (nm)
RED_NM = (650.0, 670.0)
NIR_NM = (850.0, 870.0)
CAI_NM = {"r2030": (2020.0, 2040.0), "r2100": (2090.0, 2110.0), "r2210": (2200.0, 2220.0)}
ALBEDO_NM = ((1000.0, 1300.0), (1500.0, 1750.0), (2050.0, 2300.0))
