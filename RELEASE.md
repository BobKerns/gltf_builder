# Release process

* Create a feature branch for the release.
  * No branch names are enforced, but `release/vx.x.x` would be a good choice.
* Increment the version number in [pyproject.toml](pyproject.toml).
* Update CHANGES.md
* Commit and push
* Create a PR for the release
* PR's must pass the test suite before merging.
* Create a release (e.g. through the GitHub UI) with tag in the form `vn.m.p` and a description.
* This will trigger a test deployment to (test.pypi.org)
* If that succeeds, merge the PR to `main`.
* To release to PyPi, manually trigger the workflow.
