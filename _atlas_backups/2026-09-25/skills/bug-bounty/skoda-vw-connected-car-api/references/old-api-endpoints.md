# Old VW-Group API Endpoints (Legacy Cars)

Base URL: `https://msg.volkswagen.de`
BRAND: skoda
COUNTRY: CZ

## API Endpoints (fs-car paths)

These are used by older Skoda models that don't use the modern Skoda Native API.

| Service | Path | Description |
|---------|------|-------------|
| Charger | `fs-car/bs/batterycharge/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/charger` | Battery charge status |
| Climater | `fs-car/bs/climatisation/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/climater` | Climate control status |
| Timer | `fs-car/bs/departuretimer/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/timer` | Departure timers |
| Position | `fs-car/bs/cf/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/position` | GPS position |
| Trip | `fs-car/bs/tripstatistics/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/tripdata/shortTerm` | Trip statistics |
| Heater | `fs-car/bs/rs/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/status` | Parking heater |
| Refresh | `fs-car/bs/vsr/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/requests` | Force data refresh |

## Operation List

`{homeregion}/api/rolesrights/operationlist/v3/vehicles/{vin}`

Returns available services and their status (enabled/disabled) with license expiration.

## Action Status Endpoints

| Service | Status URL |
|---------|-----------|
| Climatisation | `fs-car/bs/climatisation/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/climater/actions/{id}` |
| Battery charge | `fs-car/bs/batterycharge/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/charger/actions/{id}` |
| Departure timer | `fs-car/bs/departuretimer/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/timer/actions/{id}` |
| VSR | `fs-car/bs/vsr/v1/{BRAND}/{COUNTRY}/vehicles/{vin}/requests/{id}/jobstatus` |

## Security PIN Endpoints

| Operation | Endpoint (relative to homeregion) |
|-----------|----------------------------------|
| LOCK | `/api/rolesrights/authorization/v2/vehicles/{vin}/services/rlu_v1/operations/LOCK/security-pin-auth-requested` |
| UNLOCK | `/api/rolesrights/authorization/v2/vehicles/{vin}/services/rlu_v1/operations/UNLOCK/security-pin-auth-requested` |
| Heating | `/api/rolesrights/authorization/v2/vehicles/{vin}/services/rheating_v1/operations/P_QSACT/security-pin-auth-requested` |
| Timer | `/api/rolesrights/authorization/v2/vehicles/{vin}/services/timerprogramming_v1/operations/P_SETTINGS_AU/security-pin-auth-requested` |
| Remote Climate | `/api/rolesrights/authorization/v2/vehicles/{vin}/services/rclima_v1/operations/P_START_CLIMA_AU/security-pin-auth-requested` |
| Complete | `/api/rolesrights/authorization/v2/security-pin-auth-completed` |

## Client IDs

| Service | Client ID | Scope |
|---------|-----------|-------|
| Connect | `7f045eee-7003-4379-9968-9355ed2adb06@apps_vw-dilab_com` | openid profile address cars email birthdate badge mbb phone driversLicense dealers profession vin mileage |
| Technical | `f9a2359a-b776-46d9-bd0c-db1904343117@apps_vw-dilab_com` | openid mbb profile |
| CABS | `0f365c6e-8fff-41e0-8b02-2733ed1fe67f@apps_vw-dilab_com` | openid profile phone we_connect_vehicles |
| DCS | `72f9d29d-aa2b-40c1-bebe-4c7683681d4c@apps_vw-dilab_com` | openid dealers profile email cars address |

## Home Region

Initial home region endpoint (before vehicle-specific):
`https://mal-1a.prd.ece.vwg-connect.com/api/cs/vds/v1/vehicles/{vin}/homeRegion`

Default: `https://msg.volkswagen.de`
