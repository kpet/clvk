
# Contributing to clvk

First of all, many thanks for having made it this far :).

Contributions are most welcome!

### What can I contribute?

There are many ways in which you can contribute to clvk, here are a few examples:

- Test OpenCL applications and report the current status as a change to the
  [supported applications documentation](docs/supported-applications.md).
- Let the authors know that you'd like a given application to work with clvk
  (even if you can't test it yourself), once again as a proposed change to
  [supported applications documentation](docs/supported-applications.md).
- Report issues (usability, integration, application compatibility,
  missing features, etc).
- Package clvk in your favourite distribution.
- Fix bugs.
- Implement missing features.
- ... or anything _you_ feel like contributing.

If you're looking for ideas, make sure to have a look at
[issues](https://github.com/kpet/clvk/issues) and
[projects](https://github.com/kpet/clvk/projects)

To avoid frustration and duplicated efforts, please don't start work on an
issue that is assigned before getting in touch via comments.

### Creating a pull request

Here's a minimalistic set of guidelines for pull requests.

#### Code formatting

clvk's source code is formatted automatically using `clang-format`. Before
creating a PR, make sure the code changes you've made are correctly formatted.
You can do this using:

```
./tests/check-format.sh && echo "All code correctly formatted."
```

If the script reports that formatting errors were found, you can reformat the
code with:

```
git-clang-format origin/main --extensions cpp,hpp
```

Code formatting will be checked automatically on PRs.

#### Tests

To avoid back and forth on PRs with _code changes_ (you don't need to if you're
only changing documentation), please make sure the following tests are passing:

```
./build/simple_test
./build/api_tests
```

