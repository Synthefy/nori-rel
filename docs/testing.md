# Tests and the contracts they protect

The public suite follows the [supported API](public-api.md). Tests are retained
for their behavior and failure coverage, without a deletion percentage or fixed
test-count target. The cleanup decisions below group comparable cases.

| Tests | Public behavior | Decision |
| --- | --- | --- |
| `tests/pql/test_parser.py`, binding/scope/relationships/tree/window tests | PQL syntax, identifiers, relationships, valid target/time combinations | Keep |
| PQL SQL/dialect/context/feature tests | SQL quoting, exact window boundaries, null semantics, historical labels, scoring filters, future-data isolation, entity-ID precision and row order | Keep |
| `tests/pql/test_vendored_grammar.py` | Pinned upstream attribution and generated-parser checksums | Keep |
| `tests/feature/` | FastDFS and NoriDFS input ownership, feature alignment, inferred/saved types, future-data invariance, optional imports, agent SQL guard, executor, scoring, session, and steering | Keep |
| `tests/pipeline/`, `tests/pql/test_pipeline_roundtrip.py` | Serialization, content identity, source/model/settings checks, input hashes, replay in another process and on fresh context | Keep; add regression for different settings sharing checkpoint bytes |
| `tests/connectors/` | Driver configuration, credential validation, warehouse snapshot pins, key checks, resource ceilings and error handling | Keep simulations and separate required live checks |
| Root credential/PostgreSQL/runtime/connection tests | Read-only transactions, real server-side streaming, bounded extraction, sanitized exception chains, pool reuse/cleanup | Keep; add regressions for streaming and credential boundaries |
| `tests/run/`, PQL run-stage/connection-lifecycle tests | Event ordering, background results, timeout, pause/resume/cancel, callback failures and client shutdown | Keep |
| Root checkpoint/local-inference tests and PQL client/inference tests | Local/hosted isolation, checkpoint integrity, predictions, output validation and transport contracts | Keep deterministic tests; require real local/hosted certification for releases |
| `tests/test_legacy_adapter.py` | Deprecated aggregate-request migration behavior | Keep while the bridge remains public |
| `tests/test_release_validation.py` and package validation scripts | Protected main/tag/version checks, changelog entry for the tagged version, exact candidate identity, stale/fork rejection, installed artifact boundaries | Keep behavior checks and package validation |
| `tests/test_feature_search.py`, `tests/pql/test_feature_backends.py` | Feature-backend selection, agent-search settings, saved-program replay, PostgreSQL with NoriDFS or an agent, and hosted transport, offline with fake agents | Keep |
| RelArena, fine-tuning, Modal, and live agent end-to-end tests | Contracts of research implementations outside this distribution | Preserve with their code in the private internal repository |
| `tests/test_skills.py`, `tests/test_skill_scripts.py` | Agent skills: frontmatter, links and plugin manifests; each skill script end to end on PostgreSQL with a stand-in regressor, and on cached Nori 1.5 weights with `-m live_nori` | Keep; see the [skills README](../skills/README.md#how-the-skills-are-tested) |
| Old workflow-routing and saved-notebook-output assertions | Mirrored retired draft/stacked-PR routing and research notebook output | Remove from `main`; use workflow validation and execution of the synthetic example |

Historical research tests remain with their implementations in the private
internal repository. Promoted patches must include relevant public regression
tests; see [CONTRIBUTING.md](../CONTRIBUTING.md).
Shared fixtures and child-process helpers remain when the public tests need them.

## Run the public suite

Use a disposable PostgreSQL database and set `NORI_REL_TEST_DATABASE_URL`
privately. Integration fixtures create and remove their own schemas. The normal
suite uses deterministic predictors and does not download model weights:

```bash
uv sync --group dev --group cpu \
  --extra database --extra mysql --extra snowflake --extra databricks \
  --extra local --extra agentic
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not warehouse and not live_nori"
```

Without PostgreSQL configuration, database tests skip and that run does not
certify database behavior. CI on pull requests and main is one job,
`release-readiness`: it runs these commands on Python 3.12 against a
disposable PostgreSQL 18 and MySQL 8.4, builds the distributions, installs the wheel with
every extra outside the source checkout, and audits dependencies. The release
matrix (`release-matrix.yml`, run by `publish.yml` and on demand) installs each
extra alone on both supported Python versions, the sdist, macOS and Windows, and
tests against PostgreSQL 16 as well.

### MySQL

`tests/connectors/test_mysql_live.py` runs the shared label fixture, MySQL
types and time zones, static targets, sessions, limits, quoted names, TLS
modes and a PostgreSQL parity check against a disposable MySQL 8 server. Set
`NORI_REL_TEST_MYSQL_URL` to a user that can create databases; each test
creates and drops its own. `NORI_REL_TEST_MYSQL_CA`, the server's CA
certificate, enables the certificate-verification tests. For example:

```bash
docker run -d --name nori-rel-mysql -e MYSQL_ROOT_PASSWORD=integration \
  -p 127.0.0.1:3306:3306 mysql:8.4
export NORI_REL_TEST_MYSQL_URL=mysql://root:integration@127.0.0.1:3306/mysql
docker cp nori-rel-mysql:/var/lib/mysql/ca.pem ./mysql-ca.pem
export NORI_REL_TEST_MYSQL_CA=$PWD/mysql-ca.pem
uv run --no-sync pytest tests/connectors/test_mysql_live.py
```

Pointing the URL at the local end of an SSH tunnel runs the same suite through
the tunnel. CI runs it against a MySQL 8.4 service in `release-readiness` and
in the release matrix.

## Required release certification

Populate the selected local model cache, supply the PostgreSQL URL, and run:

```bash
uv run --no-sync pytest \
  tests/pql/test_local_postgres.py::test_offline_postgres_to_cached_local_nori \
  --require-local-model
```

The agent skills' scripts run on the same cached weights with:

```bash
uv run --no-sync pytest tests/test_skill_scripts.py -m live_nori --require-local-model
```

For real hosted inference, set `NORI_REL_TEST_HOSTED_API_KEY` privately and use:

```bash
uv run --no-sync pytest \
  tests/pql/test_client_e2e.py::test_live_hosted_pql_smoke \
  --require-hosted-model
```

Configure both warehouses with separate writer and read-only identities, then:

```bash
uv run --no-sync pytest tests/connectors/test_warehouse_live.py \
  -m "warehouse and not live_nori" --require-warehouse
```

The [warehouse guide](warehouse-connectors.md) describes configuration, parity,
owned-resource cleanup, and optional real model checks on warehouse sources.
The [release workflow](../RELEASING.md) makes both warehouse and real local/hosted
certification mandatory. Record service/model versions and reports for the
candidate; a skip or simulated predictor cannot replace live certification.

The [synthetic PostgreSQL example](../examples/README.md) additionally exercises
real prediction, pipeline save/reload through a fresh client, context/feature
hash equality, and cleanup. This checks the documented user flow; it does not
measure predictive quality on a real customer dataset.
