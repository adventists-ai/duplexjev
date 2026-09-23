# Examples

`make_examples.py` answers a fixed set of questions for a few audio clips with a released checkpoint, in one batched
forward pass per clip, and writes `docs/examples.json` for the [project page](https://adventists-ai.github.io/duplexjev/).

```bash
python make_examples.py --ckpt <checkpoint> --clips examples_config.example.json --out ../docs
```

Use only clips whose license allows redistribution. A minimal inference example will be added with the weights.
