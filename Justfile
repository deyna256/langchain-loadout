# Run the test suite.
test:
    uv run --all-extras pytest

# Check formatting, style and import order. Changes nothing; this is what CI runs.
lint:
    uv run ruff format --check .
    uv run ruff check .

# Fix what can be fixed: formatting first, then the style fixes ruff makes on its own.
format:
    uv run ruff format .
    uv run ruff check --fix .

# Check types. Optional dependencies are installed so provider adapters resolve.
type:
    uv run --all-extras ty check src

# Build the wheel and source distribution into dist/.
build:
    uv build
