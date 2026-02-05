# CLAUDE.md — nngpu

This file provides guidance for AI assistants working in this repository.

## Project Overview

**nngpu** is a new, early-stage project. The repository is currently being initialized and does not yet contain source code, build configuration, or tests.

## Repository Status

- **State**: Empty / initial setup
- **Remote**: `origin` at `switcc/nngpu`
- **Primary language**: TBD (update once source files are added)

## Directory Structure

```
nngpu/
├── CLAUDE.md          # This file — AI assistant guide
└── (no other files yet)
```

> Update this section as the project grows.

## Build & Run

No build system is configured yet. Update this section when a build system (CMake, Make, Cargo, etc.) is introduced.

```sh
# placeholder — replace with actual build commands
# make
# cmake --build build/
```

## Testing

No test framework is configured yet. Update this section when tests are added.

```sh
# placeholder — replace with actual test commands
# make test
# ctest --test-dir build/
```

## Development Workflow

### Branching

- Feature branches follow the pattern `claude/<description>-<session-id>`
- Develop on your assigned branch; do not push directly to `main`

### Commits

- Write clear, descriptive commit messages
- GPG signing is enabled and required (SSH format)
- Keep commits focused — one logical change per commit

### Code Style

Document language-specific style guidelines here once the primary language is chosen.

## Key Conventions

1. **Keep this file up to date** — When adding new modules, build steps, or conventions, update CLAUDE.md so future AI sessions have accurate context.
2. **Prefer simplicity** — Avoid over-engineering; add complexity only when justified by requirements.
3. **Test before pushing** — Once a test suite exists, run it before every push.
4. **No secrets in the repo** — Never commit API keys, credentials, or `.env` files.

## Architecture

No architecture decisions have been made yet. Document major design choices here as they arise (e.g., library vs. application, GPU backend selection, data flow patterns).

## Dependencies

No dependencies configured yet. Document dependency management approach here once established (e.g., package manager, vendored libs, system requirements).

## Useful Commands

| Task | Command |
|------|---------|
| Build | TBD |
| Test | TBD |
| Lint | TBD |
| Clean | TBD |

> Fill in this table as tooling is added.
