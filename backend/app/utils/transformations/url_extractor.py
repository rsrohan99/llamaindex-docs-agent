from os.path import relpath, splitext

from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field

class URLExtractor(TransformComponent):
  """Upsert the new docs to the docstore."""

  data_path: str = Field(
    default='./data',
    description='Relative Data directory'
  )

  def __call__(self, nodes, **kwargs):
    for node in nodes:
      node.metadata["file_path"], _ = splitext(relpath(
        node.metadata['file_path'], self.data_path+"/docs"
      ))
      # print(node.metadata)
    return nodes