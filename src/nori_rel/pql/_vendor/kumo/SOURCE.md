# Vendored Kumo PQL grammar

Source: https://github.com/kumo-ai/kumo-relational-client
Commit: `9fcdc8a111813e67787ce143fa0fdf5871d6d8de`
Directory: `packages/kumo-relational-engine/src/kumo_relational_engine/pql/grammar/`
License: Apache-2.0; package LICENSE and NOTICE are included unchanged.
Generator and runtime: ANTLR 4.13.2.

`PQLGrammar.g4` here is the pinned upstream grammar, unmodified. It is a
reference, not the grammar Nori runs: the runtime grammar is
`../../grammar/PQLGrammar.g4`, derived from this one. Diff the two to see
Synthefy's delta.

To regenerate the parser after editing the runtime grammar, from
`src/nori_rel/pql/grammar/` with the ANTLR 4.13.2 complete JAR (the version
must match `antlr4-python3-runtime` in `pyproject.toml`):

```bash
java -jar antlr-4.13.2-complete.jar -Dlanguage=Python3 -no-listener -o ../_generated PQLGrammar.g4
rm ../_generated/*.tokens ../_generated/*.interp
for f in ../_generated/PQLGrammarLexer.py ../_generated/PQLGrammarParser.py; do
  printf '%s\n%s\n' "# Derived from NVIDIA's Apache-2.0 PQL grammar; modified by Synthefy." \
    "# See ../_vendor/kumo/LICENSE, NOTICE and ../grammar/PQLGrammar.g4." | cat - "$f" > "$f.new" && mv "$f.new" "$f"
done
(cd .. && shasum -a 256 grammar/PQLGrammar.g4 _generated/PQLGrammarLexer.py _generated/PQLGrammarParser.py > grammar/CHECKSUMS.txt)
```

`tests/pql/test_vendored_grammar.py` asserts those checksums, so a grammar
edit without regeneration fails the ordinary test job.

Upstream's generated `PQLGrammarLexer.py`, `PQLGrammarParser.py` and
`PQLGrammarVisitor.py` were removed once nothing imported them. They are
machine output reproducible from the `.g4` above, so keeping a second stale
copy of a parser Nori does not run had no reader and no purpose.

Original file SHA-256 hashes, asserted by `tests/pql/test_vendored_grammar.py`:

```text
9de5d1ee6fdf2de2675bf866d81f4617515e4d0dcbbd2fdcadf5ecc0edbb7ea6  PQLGrammar.g4
8807a999af23d76b050f71bcb8dc9d5cb68e7c455d4eafa3855e95bff9100fc6  LICENSE
fcca034cbf6770625d0789b75acb3aba179c16bf9a3a754a3095f8fb282ee88d  NOTICE
```
