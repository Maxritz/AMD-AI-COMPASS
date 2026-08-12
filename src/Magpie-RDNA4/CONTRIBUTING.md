# Contributing to Magpie

Thanks for your interest in Magpie! This guide explains how to contribute, report issues, and submit changes.

## Before You Start

- Read `README.md` to understand the project scope and usage.
- Ensure you have a supported GPU environment (AMD HIP or NVIDIA CUDA), or use Docker.

## Development Setup

```bash
# Create virtualenv and install dependencies
make install-dev

# Run basic checks
make verify
```

## Workflow

1. Create a new branch from `main`.
2. Keep changes focused and scoped.
3. Run lint checks before submitting:

```bash
make lint
```

4. Open a Pull Request with motivation, impact, and verification steps.

## Code Style and Quality

- Lint: `ruff`
- Formatting: `ruff`
- Add documentation or comments when needed.

## Testing and Verification

This project may depend on GPU hardware or drivers. In your PR, include:

- Test environment (GPU model, driver version, HIP/CUDA version)
- Execution mode (local or Docker)
- Key commands and output summary (e.g., `python -m Magpie analyze ...`)

## Filing Issues

Please include:

- Reproduction steps
- Expected vs actual behavior
- Environment (OS, GPU, driver, Python version)
- Relevant logs or a minimal repro

## Security

If you discover a security issue, do not open a public issue. Contact maintainers through a private channel.

## Suggested Contributions

- Add evaluation modes or kernel config examples
- Fix edge cases in the evaluation pipeline
- Improve report structure or readability
- Improve docs, examples, and tests

## License

By contributing, you agree that your contributions are licensed under the repository `LICENSE` (MIT).

