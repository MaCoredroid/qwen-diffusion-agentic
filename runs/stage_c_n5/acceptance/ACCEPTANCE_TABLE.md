# Runtime-alignment ACCEPTANCE GATE

Scripted (NO model) in-episode checks INSIDE the official per-instance swebench image: `import <pkg>` + the exact instance test command, via the driver's container mechanism (workspace seeded from the image /testbed, bind-mounted, shell routed through `docker exec` with conda `testbed` activation).

**Result: 5/5 accepted**

| instance | pkg | import rc | tests ran | dep-error | accepted |
|---|---|---|---|---|---|
| django__django-11119 | django | 0 | yes | none | PASS |
| django__django-12754 | django | 0 | yes | none | PASS |
| django__django-13741 | django | 0 | yes | none | PASS |
| pytest-dev__pytest-8399 | pytest | 0 | yes | none | PASS |
| sympy__sympy-13757 | sympy | 0 | yes | none | PASS |

Evidence: `acceptance_results.json`, `<instance>__import.log`, `<instance>__test.log` in this directory.
