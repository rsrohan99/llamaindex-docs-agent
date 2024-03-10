"""Custom Markdown node parser."""
import re
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, TextNode
from llama_index.readers.file import MarkdownReader

def generate_markdown_header_id(header_text, previous_ids=None):
  if previous_ids is None:
    previous_ids = set()

  # Step 1: Convert to lowercase
  id = header_text.lower()

  # Step 2: Remove non-word text
  id = re.sub(r'\W+', ' ', id)

  # Step 3: Convert spaces to hyphens
  id = id.strip().replace(' ', '-')

  # Step 4: Remove multiple consecutive hyphens
  id = re.sub(r'-+', '-', id)

  # Step 5 & 6: Ensure uniqueness
  original_id = id
  counter = 1
  while id in previous_ids:
    id = f"{original_id}-{counter}"
    counter += 1

  # Add the new unique id to the set of previous ids
  # previous_ids.add(id)

  return id

class CustomMarkdownNodeParser(MarkdownNodeParser):
    """Markdown node parser.

    Splits a document into Nodes using custom Markdown splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    def _update_metadata(
        self, headers_metadata: dict, new_header: str, new_header_level: int
    ) -> dict:
        """Update the markdown headers for metadata.

        Removes all headers that are equal or less than the level
        of the newly found header
        """
        updated_headers = {}

        for i in range(1, new_header_level):
            key = f"Header {i}"
            if key in headers_metadata:
                updated_headers[key] = headers_metadata[key]

        updated_headers[f"Header {new_header_level}"] = new_header
        # print(updated_headers)
        updated_headers['section_link'] \
            = generate_markdown_header_id(new_header, updated_headers.values())
        return updated_headers

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""
        md_reader = MarkdownReader()

        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        node.text = md_reader.remove_hyperlinks(node.text)
        node.text = md_reader.remove_images(node.text)

        if self.include_metadata:
            node.metadata = {**node.metadata, **metadata}

        return node