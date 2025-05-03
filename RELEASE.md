# Release process

* Create a feature branch for the release.
  * No branch names are enforced, but `release/vx.x.x` would be a good choice.
* Increment the version number in [pyproject.toml](pyproject.toml).
* Update [CHANGES.md](CHANGES.md)
* If any `*.mmd` files have changed, execute `npm run build`
* Commit, including changed `.svg` files and push
* [Create a PR](https://github.com/BobKerns/gltf_builder/pulls) for the release
* PR's must pass the test suite before merging. Monitor the workflow on the [GitHub Actions page for the project](https://github.com/BobKerns/gltf_builder/actions)
* Create a release (e.g. [through the GitHub UI](https://github.com/BobKerns/gltf_builder/releases)) with tag in the form `vn.m.p` and a description.
* This will trigger a test deployment to ([test.pypi.org](https://test.pypi.org/project/gltf-builder/))
* If that succeeds, merge the PR to `main`.
* To release to PyPi, manually trigger the workflow.
* If tests and checks pass, this will publish to [pypi.org](https://pypi.org/project/gltf-builder/).

