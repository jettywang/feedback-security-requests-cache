# Tests
See [Contributing Guide](https://requests-cache.readthedocs.io/en/main/contributing.html#testing)
for details on unit and integration tests.

## Contract tests (`tests/contract/`)

Backend-agnostic "anti-miswrite" contract tests. The same suite is mixed into one
test class per backend (in-memory and temporary-file SQLite today) and observes real
backend reads/writes plus the returned response, to guarantee a response is never
persisted when:

1. The cache is disabled for the request (`cache_disabled()`).
2. The request/response is marked non-storable (`Cache-Control: no-store`).
3. A `filter_fn` rejects the response.

To verify the suite actually catches a broken refusal gate, set
`REQUESTS_CACHE_FAIL_OPEN=1`. This holds the write gate open, so each enforced
clause fails with a message naming the exact rule breached (and the cache key that
was wrongly written). Leave it unset for normal CI runs:

```bash
pytest tests/contract/                                  # strict mode (default)
REQUESTS_CACHE_FAIL_OPEN=1 pytest tests/contract/       # tripwire / clause attribution
```

