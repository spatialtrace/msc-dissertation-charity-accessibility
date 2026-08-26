# Data availability and licensing

This repository distributes code and aggregate analytical summaries. It does not redistribute the underlying datasets.

## Public or independently obtainable sources

- UK Census and LSOA statistics: Office for National Statistics/Nomis. Reuse is subject to the Open Government Licence and source attribution.
- LSOA, ICB and rural–urban geography: Office for National Statistics/Open Geography, subject to the terms stated with each download.
- Charity register and annual-return records: Charity Commission for England and Wales.
- Company records: Companies House.
- Third Sector and Civil Society Sector database financial spine: UK Third Sector Database, CC BY 4.0.
- Inflation adjustment: ONS CPIH all-items annual series, 2015 = 100.

The code documents the variables and transformations applied to these sources. Users must obtain current copies from the relevant provider and comply with their terms.

## Restricted Ordnance Survey material

The road analysis uses December 2021 OS MasterMap Highways data obtained through EDINA Digimap. The source GPKG, extracted RoadLink geometry, routing graphs, node arrays, OD caches and georeferenced derivative layers are excluded because they must not be made available to unauthorised third parties.

Non-georeferenced raster maps derived for the dissertation require the following acknowledgement:

> © Crown copyright and database rights 2026 Ordnance Survey (AC0000851941)

No Ordnance Survey source or reversible road-network data is included here.

## Charity-level and fine-grained analytical records

The following are also excluded from the repository:

- charity names, registered addresses, postcodes and coordinates;
- charity-level income and address-provenance tables;
- complete annual LSOA accessibility and HP–LA status files;
- LSOA trajectory tables and georeferenced GIS outputs.

Aggregate regional, ICB, settlement and method-QA summaries are included under [`results`](results).

## Useful source links

- [Nomis copyright and licensing](https://www.nomisweb.co.uk/home/copyright.asp)
- [ONS Open Geography](https://www.ons.gov.uk/methodology/geography/geographicalproducts/opengeography)
- [UK Third Sector Database](https://uk-third-sector-database.github.io/data/)
- [TCSS charity financial-record guidance](https://uk-third-sector-database.github.io/guidance/tcss-charity-financial-records-guidance.html)
- [EDINA Digimap licensing FAQ](https://digimap.edina.ac.uk/help/faqs/licensing/)
