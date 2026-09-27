# configs

| file | who owns it | what it controls |
|---|---|---|
| `whitelist.yaml` | engineer | the only parameters and spatial forms any hypothesis may use |
| `checks.yaml` | engineer | physics validation criteria, device spec, replan stop rules |
| `pipeline.yaml` | operator | data source, device models, hypothesis mode, optimiser, gate mode |

`whitelist.yaml` and `checks.yaml` define what counts as a valid answer. They are
never written by a model. Widening either is a deliberate engineering decision,
recorded in version control.
