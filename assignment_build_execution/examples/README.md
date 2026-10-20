# Example projects

Three small projects that show the `assignment_config.yaml` features. Each one is a
folder holding `build.sh` / `run.sh` scripts, plus a config file. Most use a `src/`
subfolder; `multi-module/` puts each module's scripts in a folder named after it.

| Project          | Shows                                                                        |
|------------------|------------------------------------------------------------------------------|
| `greeter/`       | `run.sh` that reads input from stdin. Configurations supply their own input using `stdin:`, alone or together with `args:`. |
| `build-matrix/`  | Several build configurations in one list. One fails on purpose, and the rest still run. |
| `multi-module/`  | Two modules (`alpha`, `beta`) in one `modules:` list. Pick one with `-m alpha` or `-m beta`; there is no `current` module, so `-m` is required. |

The runner only treats folders containing `.git` as projects, and the examples are not
committed as git repositories. To try one, copy it somewhere and initialise git there:

```bash
cp -R examples/greeter /tmp/greeter-demo        # or any other location
cd /tmp/greeter-demo
git init -q -b main && git add -A && git -c user.name=me -c user.email=me@example.com commit -qm init
/path/to/assignment-runner/run_assignments.sh
cat assignment_output.txt
```

`tests/run_tests.sh` does the same copying and checks the results automatically.
