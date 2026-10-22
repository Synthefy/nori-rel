# Attribution

Original Synthefy code in Nori-Rel is distributed under the [MIT License](LICENSE).
Third-party code retains its own license. The project license does not grant
rights to third-party model weights, datasets, or hosted services.

## PQL grammar and generated parser

The PQL grammar derives from the NVIDIA Kumo Relational Client at commit
`9fcdc8a111813e67787ce143fa0fdf5871d6d8de`:

- Source: <https://github.com/kumo-ai/kumo-relational-client>
- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
- License: Apache-2.0.
- Unmodified grammar, license, notice, and provenance:
  [`src/nori_rel/pql/_vendor/kumo/`](src/nori_rel/pql/_vendor/kumo/).
- Modified grammar: [`src/nori_rel/pql/grammar/PQLGrammar.g4`](src/nori_rel/pql/grammar/PQLGrammar.g4).
- Derived parser: [`src/nori_rel/pql/_generated/`](src/nori_rel/pql/_generated/).

The modified grammar and generated parser retain this Apache-2.0 notice.
The pinned upstream license and notice are included unchanged. The upstream
notice covers its full distribution; it does not imply that Nori-Rel ships every
component mentioned there. Parser generation instructions and source checksums
are in the vendored `SOURCE.md`.

## Installed dependencies

[FastDFS](https://github.com/HKUSHXLab/fastdfs) supplies the supported relational
feature implementation under Apache-2.0. It uses
[Featuretools](https://github.com/alteryx/featuretools), copyright Feature Labs,
Inc., under BSD-3-Clause. These libraries are installed as dependencies and are
not vendored in Nori-Rel. ANTLR's Python runtime is separately installed under
BSD-3-Clause. Other dependencies retain the terms in their distributions.

The local Nori runtime and downloaded model artifacts are separately distributed.
Review the selected model's license and model card before redistributing weights.

## Examples and research

The examples on `main` generate synthetic data; they do not redistribute a
research dataset. The Colab quickstart in `colab/` downloads RelBench's prepared
[`rel-f1`](https://relbench.stanford.edu/datasets/rel-f1/) database from Stanford
when it runs. That data is not included in this repository or its packages, and
it keeps the terms of its source. Research code, attribution, methods, and results are maintained
privately and are outside the supported release package.
