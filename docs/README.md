build the docs

```bash
cd docs
make docs
```

clean the docs

```bash
cd docs
make clean
```

preview the docs using python's built-in server

```bash
cd docs
python -m http.server --directory build/html
```
