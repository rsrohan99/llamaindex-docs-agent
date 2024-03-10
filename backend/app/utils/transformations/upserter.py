import os

from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field
from llama_index.core.storage.docstore.types import BaseDocumentStore

class Upserter(TransformComponent):
  """Extract the relative path or a documentation page."""

  docstore: BaseDocumentStore = Field(
    description='Document store to check for duplicates'
  )

  persist_dir: str = Field(
    description='persist path for docstore'
  )

  def __call__(self, nodes, **kwargs):
    assert self.docstore is not None

    self.docstore.add_documents(nodes)
    self.docstore.persist(
      persist_path=os.path.join(self.persist_dir, "docstore.json")
    )
    return nodes