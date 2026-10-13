---
title: OreSight
emoji: 🛰️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# OreSight

Copper prospectivity from satellite data: alteration mapping (EMIT, ASTER), lineaments, CPI, ranked prospective zones and suggested field-check locations.

The app asks for a user name and password. New analyses are saved to a private dataset, so they survive restarts.

Configuration (Space settings → Variables and secrets):

| Name | Kind | Purpose |
|---|---|---|
| `GEOSCAN_PASSWORD` | secret | Login password (user name: `geoscan`, or `GEOSCAN_USER`) |
| `HF_TOKEN` | secret | Token with write access to the storage dataset |
| `GEOSCAN_STORE_REPO` | variable | Private dataset that holds data and results, e.g. `user/geoscan-store` |
| `GEE_SERVICE_ACCOUNT_KEY` | secret | JSON key of an Earth Engine service account |
