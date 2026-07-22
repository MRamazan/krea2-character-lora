# Implementation Tasks

## Phase 1: Inventory

- Map all source notebook behavior to target modules.
- Identify project-specific generalizations that must be removed.
- Identify exact character-training defaults and third-party schema constraints.

## Phase 2: Core package

- Implement paths, errors, configuration, manifests, and public report types.
- Implement authentication and environment preparation.
- Implement training and inference asset resolution.

## Phase 3: Dataset

- Implement extraction, canonical pair discovery, validation, trigger audit, fingerprinting, duplicate detection, gallery rendering, and issue reporting.

## Phase 4: Training

- Implement AI Toolkit configuration generation, preflight, smoke training, production training, resume, run manifests, and checkpoint inventory.

## Phase 5: Evaluation

- Implement inference helpers, non-destructive adapter loading, base comparison, checkpoint sweep, scale sweep, manual selection, and evaluation manifests.

## Phase 6: Export

- Implement dynamic packaging, hash inventory, package freeze, and optional Colab download.

## Phase 7: Notebook and validation

- Finalize the four notebook cells.
- Expand contract and behavior tests.
- Run local checks.
- Document Colab smoke validation results.
