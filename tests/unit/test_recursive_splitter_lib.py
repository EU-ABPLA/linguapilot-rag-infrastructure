from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def test_factory_routes_recursive_provider() -> None:
	instance = SplitterFactory.create({"splitter": {"provider": "recursive"}})
	assert isinstance(instance, RecursiveSplitter)


def test_recursive_splitter_splits_markdown_by_headings() -> None:
	text = (
		"# Title\n\n"
		"Intro paragraph.\n\n"
		"## Section A\n"
		"Alpha content.\n\n"
		"## Section B\n"
		"Beta content."
	)
	splitter = RecursiveSplitter(chunk_size=48, chunk_overlap=0)
	chunks = splitter.split_text(text)
	assert len(chunks) >= 2
	assert any("## Section A" in chunk for chunk in chunks)
	assert any("## Section B" in chunk for chunk in chunks)


def test_recursive_splitter_keeps_code_block_intact_when_chunk_allows() -> None:
	text = (
		"# Title\n\n"
		"```python\n"
		"def add(a, b):\n"
		"    return a + b\n"
		"```\n\n"
		"Tail."
	)
	splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
	chunks = splitter.split_text(text)
	code_chunks = [chunk for chunk in chunks if "```python" in chunk]
	assert len(code_chunks) == 1
	assert "```" in code_chunks[0]


def test_recursive_splitter_validates_input() -> None:
	splitter = RecursiveSplitter()
	try:
		splitter.split_text(123)
		assert False
	except ValueError as exc:
		assert "recursive validation error" in str(exc)
