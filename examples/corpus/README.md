# Nimbus corpus

The documentation of **Nimbus**, a fictional cloud-storage service: 200 documents (Spanish
primary, ~20 % English) written to make retrieval hard in specific, labeled ways, plus 60
hand-labeled questions. It exists so the benchmark can show *when each RAG pattern wins and
why*, not only that it runs.

```bash
python examples/corpus/generate.py            # rewrites corpus.jsonl (same bytes every time)
python examples/corpus/generate.py out.jsonl  # or anywhere else
```

`generate.py` is stdlib only and seeded (`random.Random(20261014)`); the text is hand-written
in the script and the seed only picks headings, layouts and release-note bullets. A test
checks that regenerating gives the committed bytes.

## Documents (`corpus.jsonl`)

Each line: `{"id", "text", "metadata": {"type", "trap", "lang", "version"?}}`.

| Family | Count | `trap` | What makes it hard |
|---|---|---|---|
| Error articles `err-NNNN` | 40 (31 es / 9 en) | `exact` | 15 sibling pairs (`ERR-4012`/`ERR-4013`) share almost every word; only the code and the deciding detail differ. |
| Plans `sku-NNx` | 20 (16 es / 4 en) | `exact` | Monthly/annual and small/large siblings (`SKU-52M`/`SKU-52A`) differ only in limits and price. |
| Billing / account FAQs `faq-*` | 50 (40 es / 10 en) | `paraphrase` | Formal vocabulary ("cargo duplicado", "baja de la suscripción") that colloquial questions do not use. |
| Policies `pol-*-2024` / `-2026` | 2 × 15 (24 es / 6 en) | `distractor` | Same title and text; only the numbers differ. 2024 says "Reemplazada por la versión 2026". |
| Release notes `rn-*`, blog posts `blog-*` | 60 (48 es / 12 en) | `noise` | Reuse product words and even error codes ("mejoramos el mensaje de ERR-4012") without answering anything. |

## Questions (`queries.jsonl`)

Written by hand after reading the corpus — not generated. Each line:
`{"id", "query", "relevant": [doc ids], "type", "lang"}`.

| `type` | Count | Example | Hypothesis |
|---|---|---|---|
| `exact` | 15 | "¿Qué significa ERR-4012?" | `hybrid` wins: BM25 matches the code token |
| `paraphrase` | 20 | "me cobraron dos veces este mes" | `multi_query` / `hyde` win: the rewrite bridges vocabulary |
| `distractor` | 15 | "¿Cuántos días tengo para pedir que me devuelvan la plata?" | a reranker helps tell versions apart |
| `noise` | 10 | "¿Cómo activo la verificación en dos pasos?" | a reranker pushes the look-alike posts down |

13 of 60 questions are in English.

### Labeling rules

- `relevant` lists the documents that **directly answer** the question. A document that
  only mentions the topic, or from which the answer could be derived (an annual price next to
  a monthly one), is not labeled.
- When more than one document answers, all are listed (e.g. "quiero meter más gigas" →
  the add-on FAQ and both storage add-on SKUs; an English and a Spanish FAQ that say the same
  thing).
- `distractor` questions count **only the 2026 version** as relevant: retrieving the
  superseded 2024 policy gives a wrong answer with the same words.
- `noise` questions have real answers in the corpus; the noise is the posts around them.
- Paraphrase questions were checked against their target FAQ for shared content words; the
  only overlaps left are generic ("cuenta", "plan", "pago").

The hypotheses above are what the benchmark tests, not what it assumes. Results — including
where a hypothesis fails — are reported from `examples/benchmark.py` runs.
