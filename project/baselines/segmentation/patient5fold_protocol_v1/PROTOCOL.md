# RVO-ME Patient-Disjoint Baseline Protocol v1

This protocol is a canonical copy of the already frozen `patient_5fold_v1` split. It does not create or optimize a new patient split.

## Required experimental rule

All conventional baselines and the proposed geometry-aware model must use these exact patient/image roles for the primary Track-B five-fold comparison.

## Fold counts

| Fold | Train patients | Val patients | Test patients | Train images | Val images | Test images |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91 | 13 | 26 | 2110 | 303 | 599 |
| 2 | 91 | 13 | 26 | 2136 | 270 | 606 |
| 3 | 90 | 14 | 26 | 2106 | 308 | 598 |
| 4 | 89 | 15 | 26 | 2039 | 342 | 631 |
| 5 | 89 | 15 | 26 | 2072 | 362 | 578 |

## Fingerprints

- Frozen source split SHA-256: `7e7c2ddf51054bf998b2d750b50c74983b9bc5b751b762213bc96343ffe01346`
- Canonical protocol SHA-256: `52ed516dcc21f1c51068c59fb662c9a0030b143e6440d6f4a0380df713c9bcaf`

Do not edit these files after model experiments begin. Any future split variant must use a new protocol version.
