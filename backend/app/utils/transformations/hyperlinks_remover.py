from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field
from llama_index.readers.file import MarkdownReader

class HyperlinksRemover(TransformComponent):
  """Remove hyperlinks and images from md or mdx."""

  def __call__(self, nodes, **kwargs):
    md_reader = MarkdownReader()
    for node in nodes:
      node.text = md_reader.remove_hyperlinks(node.text)
      node.text = md_reader.remove_images(node.text)
      # print(node.metadata)
    return nodes