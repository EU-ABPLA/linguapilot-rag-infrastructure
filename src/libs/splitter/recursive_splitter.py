from typing import Any, List, Optional, Sequence

from libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
	def __init__(
		self,
		chunk_size: int = 800,
		chunk_overlap: int = 100,
		separators: Optional[Sequence[str]] = None,
	):
		self.chunk_size = max(1, int(chunk_size))
		self.chunk_overlap = max(0, int(chunk_overlap))
		if separators is None:
			self.separators = (
				"\n```",
				"\n# ",
				"\n## ",
				"\n### ",
				"\n\n",
				"\n",
				" ",
			)
		else:
			self.separators = tuple(separators)

	def split_text(self, text: str, trace: Optional[Any] = None) -> List[str]:
		if not isinstance(text, str):
			raise ValueError("recursive validation error: text must be a string")
		if not text.strip():
			return []
		return self._recursive_split(text, list(self.separators))

	def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
		if len(text) <= self.chunk_size:
			value = text.strip()
			return [value] if value else []
		if not separators:
			return self._hard_split(text)
		separator = separators[0]
		parts = _split_keep_separator(text, separator)
		if len(parts) <= 1:
			return self._recursive_split(text, separators[1:])
		output: List[str] = []
		buffer = ""
		for part in parts:
			if len(part) > self.chunk_size:
				if buffer.strip():
					output.append(buffer.strip())
					buffer = ""
				output.extend(self._recursive_split(part, separators[1:]))
				continue
			if not buffer:
				buffer = part
				continue
			if len(buffer) + len(part) <= self.chunk_size:
				buffer += part
				continue
			output.append(buffer.strip())
			buffer = self._attach_overlap(buffer, part)
		if buffer.strip():
			output.append(buffer.strip())
		return output

	def _hard_split(self, text: str) -> List[str]:
		step = self.chunk_size - self.chunk_overlap
		if step <= 0:
			step = self.chunk_size
		output: List[str] = []
		start = 0
		while start < len(text):
			end = min(start + self.chunk_size, len(text))
			chunk = text[start:end].strip()
			if chunk:
				output.append(chunk)
			if end >= len(text):
				break
			start += step
		return output

	def _attach_overlap(self, current: str, next_part: str) -> str:
		if self.chunk_overlap <= 0:
			return next_part
		tail = current[-self.chunk_overlap :]
		return tail + next_part


def _split_keep_separator(text: str, separator: str) -> List[str]:
	if not separator:
		return [text]
	parts = text.split(separator)
	if len(parts) <= 1:
		return [text]
	output: List[str] = []
	for index, value in enumerate(parts):
		if index == 0:
			if value:
				output.append(value)
			continue
		output.append(separator + value)
	return output
