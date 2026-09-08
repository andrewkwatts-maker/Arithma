# Arithma

<!-- GENERATED FROM GitReview/extraction/modules.yaml - DO NOT EDIT BY HAND -->

> **Crate:** `arithma_core` &nbsp;|&nbsp; **Source:** `plugins/pt-arithmos`
> **Extraction order:** 0 &nbsp;|&nbsp; **PyPI:** yes

## Purpose

The symbolic and exact-arithmetic engine. Already extracted, tested and publish-ready; it is listed here because it is the module every parametric thing depends on, not because it needs work.

## Responsibilities

- Expression AST, simplification, calculus, equation solving
- Exact integer and rational arithmetic
- Matrices, tensors, units and SI, constants
- Fourier, probability distributions, numerical methods

### Explicitly not this module's job

- Geometry evaluation in a hot path — it builds expression trees, it does not compute f32 (Topos and Stereoma do that)
- Anything GPU

## Canonical capabilities

These are registered to this module in the manifest. Each has exactly one owner
across the whole extracted set; a reappearance elsewhere is a regression.

- symbolic_math

## Dependencies

**Upstream modules**

_(none)_

**Consumers**

- stereoma
- phantasia
- plastes
- periodica
- metaphysica
- eml-math

## Open issues

- DO NOT merge the 45k LOC that pt-arithmos holds and arithma_core does not. The repo is authoritative and publish-ready; the only requirement is that PT can import arithma_core.
- Licence contradiction, owner decision: LICENSE and both manifests say MIT while 62 source files say All Rights Reserved. Blocks publication.

## Status

**Live.** Extracted, tested and in use.
