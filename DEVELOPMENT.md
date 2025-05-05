# Development

## Prerequisites

- [VCS] [git](#git-git-lfs)
- [Binary files] [git-lfs](#git-git-lfs)
- [Project management] [`uv`](#uv)
- \[For testing] [`node`, `npm`](https://nodejs.org/)
- \[Recommended] [`direnv](https://direnv.net/docs/installation.html)
- \[IDE] [`Visual Studio Code](https://code.visualstudio.com/)

### Getting started

- Clone this repo into your local environment
- Install any prerequisites you are missing.
  - Consider updating to the latest versions.
- Type `direnv allow`
  - If not using `direnv`:
    - `uv sync`
    - `npm install`
    - Examine [.envrc](.envrc) for other steps, such as adding to your PATH or other environment variables.
- Launch Visual Studio Code
  - Select the appropriate python environment (i.e. the .venv for the project).

### [devtool](scripts/devtool)

There is a python script in [scripts/devtool](scripts/devtool) that handles main development tasks. It can be run via `uv run scripts/devtool`, or if your environment is properly set up, just `devtool`.

```bash
rwk──> devtool
Usage: devtool [OPTIONS] COMMAND [ARGS]...

  Update and generate documentation and other files, and perform other
  project-specific tasks.

  Supply the --help option for more information on a specific command.

Options:
  --debug    Enable debug mode.
  --verbose  Enable verbose mode.
  --help     Show this message and exit.

Commands:
  browse  [<PATH>] Open the documentation in a browser.
  server  <CMD> Operations on the local documentation server.
  setup   Set up the project.
  sync    Update the requirements.txt and uv.lock files.
  update  <CMD> Operations that update the project.
```

`devtool browse`_`[path]`_ will start a local webserver (if it is not already running),and watch for changes, updating whenever a file is modified.

## Detailed prerequisites

### [git](<https://git-scm.com)>), [git-lfs](https://git-lfs.com)

If you have ever installed [git-lfs](https://git-lfs.com), you are all set.

If not, you will need to download and install it, and execute `git lfs install` once. You only need to do this once for each user account on each machine that you use it. It configures per-user global settings to support `git-lfs`.

### [uv](https://docs.astral.sh/uv/)

[uv](https://docs.astral.sh/uv/) is a modern, superior, and much faster replacement for `pip`, `venv`, `poetry`, various packaging tools, etc.

The familiar `pip` interface is there via `uv pip ...`, but is not generally needed.

Add project dependencies via `uv add`. Create the virtual environment and install dependencies with `uv sync`. But see the next section for how to automate this further.

### [node](https://nodejs.org/), [npm](https://nodejs.org/)

 [node](https://nodejs.org/) and [npm](https://nodejs.org/) are used only for the test suite (to run the official glTF validator), and a local dev webserver to view the rendered Markdown files.

### [direnv](https://direnv.net)

`direnv` enables automatic switching of environments when you cd into a directory. An alternative is to use `oh-my-zsh` with the `virtualenv` plugin, but it doesn't handle other tasks such as setting PATH.

With `direnv` enabled, `cd` into a project folder, and your shell environment will automatically be configured to work in that project. `cd` out, and the original settings (`$PATH`, etc.) are restored.

> [!NOTE]
> The first time you enter a project folder, you will be asked to approve the actions in [.envrc](.envrc).
>
> It is a good idea to inspect the content first, then issue the `direnv allow` command. You only need to do this once for each version of the file. If the file changes, due to an edit or due to changing branches, and the content has not been approved before you will be prompted again.

This is controlled by the [.envrc](.envrc) file. This file:

- Checks that you have the prerequisites installed (`uv`, `node`, `npm`).
- Adds the appropriate directories to the `$PATH` variable.
- Checks if the project dependencies have been installed, and if not:
  - Creates the virtual environment via `uv venv .venv`
  - Activates the virtual environment via
     `source .venv/bin/activate`.
  - does `uv run scripts/devtool setup` to do the rest of the setup.
- else:
  - Activates the environment: `source .venv/bin/activate`.

If you don't install `direnv`, you will have to perform these actions yourself, every time you start working in the project.

### [Visual Studio Code](https://code.visualstudio.com)

This includes workspace settings for VSCode optimized for this project. Items of note:

- The explorer view is configured to collapse many related files as "subfiles" of the main file. [README.md](README.md), [CONFIG](CONFIG.md), and more. See [CONFIG.md](CONFIG.md) for more information.
- A (large) set of recommended plugins is included and will be offered by VSCode when you open the project. These offer many services; most are optional but helpful. Important ones include:

  - Language plugins. These help catch errors. Currently using [ruff](https://docs.astral.sh/ruff/), as it seems to do the best job at catching type errors—though with many issues.
  - I don't use automatic formatting, because it makes many things unreadable, especially complex comprehension expressions. Instead, I try manually for consistency and readability.
    - My only rule: Make it as readable and maintainable as possible.
  - Image tools, such as the [gltTF Model Viewer](https://marketplace.visualstudio.com/items/?itemName=cloudedcat.vscode-model-viewer) and [glTF Tools](https://marketplace.visualstudio.com/items/?itemName=cesium.gltf-vscode).
  - Tools for markdown and Mermaid diagrams.
  - Tools for working with `git` and [GitHub](https://github.com/BobKerns/gltf_builder), including [git-lfs](https://git-lfs.com)locking.

### VSCode Extensions

The project recommends the following extensions grouped by category:

> AI Generated from the list of extension IDs in [.vscode/extensions.json](.vscode/extensions.json)

#### Language & Development Tools

| Extension | Description |
|-----------|-------------|
| ruff | Python linter for catching errors and enforcing code quality |
| Python | IntelliSense, linting, debugging, and code navigation for Python |
| Pylance | Fast, feature-rich language support for Python |
| TOML | TOML language support |
| YAML | YAML language support |

#### glTF & 3D Tools

| Extension | Description |
|-----------|-------------|
| glTF Model Viewer | Previewer for 3D glTF models |
| glTF Tools | Tools for validating and viewing glTF assets |
| 3D Viewer | Visualize and inspect 3D models |

#### Markdown & Documentation

| Extension | Description |
|-----------|-------------|
| Markdown All in One | Keyboard shortcuts, TOC, auto preview for Markdown |
| Mermaid Preview | Preview Mermaid diagrams in Markdown documents |
| Mermaid Graphical Editor | Create Mermaid diagrams with an interactive GUI |
| Mermaid Chart | The "official" mermaid extension from mermaidchart.com |
| vscode-mermAid | Microsoft's Copilot plugin to generate Mermaid diagrams from Copilot chats |
| Markdown Preview Enhanced | More powerful Markdown previewer |
| markdownlint | Linting and style checking for Markdown |

#### Source Control & Collaboration

| Extension | Description |
|-----------|-------------|
| GitLens | Enhances Git capabilities within VS Code |
| Git Graph | View a Git graph of your repository |
| Git LFS | Support for Git Large File Storage (LFS) |
| GitHub Pull Requests | Review and manage GitHub pull requests |

#### Productivity & UI

| Extension | Description |
|-----------|-------------|
| direnv | direnv integration for automatic environment switching |
| Todo Tree | Displays TODO comments in a tree view |
| Code Spell Checker | Spell checker for source code |
| Path Intellisense | Autocompletes filenames in imports |

## Testing

The unit tests and their organization are described in the [Testing README](testing/README.md). Files created by the tests are automatically validated using the [official Khronos validator](https://github.khronos.org/glTF-Validator/).

## Release process

See [RELEASE.md](RELEASE.md) for how to create a new release.

## The Compiler

This is basically a compiler, translating from an input representation to the final product.

A bit of documentation [here](docs/compiler.md).
