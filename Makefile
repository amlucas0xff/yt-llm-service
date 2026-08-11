.PHONY: test smoke

# The unit suite mocks every model call and stubs torch (tests/stubs/torch.py),
# so it needs the API and CLI libraries but none of the ML stack. Keeping the
# list here rather than reusing requirements.txt is what makes `make test`
# install in seconds instead of pulling several GB of CUDA wheels.
TEST_DEPS := pytest pytest-asyncio pytest-mock \
             fastapi httpx pydantic python-dotenv python-multipart \
             typer rich yt-dlp pyyaml

# sys.path, TEMP_DIR/OUTPUT_DIR and the yt-dlp shim are all set up by
# conftest.py, so this needs no environment of its own.
test:
	uv run --no-project --python 3.12 \
	  $(foreach dep,$(TEST_DEPS),--with $(dep)) \
	  pytest tests/ $(ARGS)

# End-to-end against the real containers. Slow, needs GPUs — run on demand.
smoke:
	./scripts/smoke.sh $(ARGS)
