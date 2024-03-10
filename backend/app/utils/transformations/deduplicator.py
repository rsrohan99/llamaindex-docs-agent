from typing import Callable

from llama_index.core.schema import TransformComponent
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.bridge.pydantic import Field

class Deduplicator(TransformComponent):
  """Deduplicate documents and also delete old and updated ones."""

  docstore: BaseDocumentStore = Field(
    description='Document store to check for duplicates'
  )

  cleanup_fn: Callable = Field(
    default=lambda _:...,
    description="after deleting missing nodes, call this function."
  )

  def __call__(self, nodes, **kwargs):
    assert self.docstore is not None

    existing_doc_ids_before = set(
      self.docstore.get_all_document_hashes().values()
    )
    doc_ids_from_nodes = set()
    deduped_nodes_to_run = {}

    for node in nodes:
      ref_doc_id = node.ref_doc_id if node.ref_doc_id else node.id_
      doc_ids_from_nodes.add(ref_doc_id)
      existing_hash = self.docstore.get_document_hash(ref_doc_id)
      if not existing_hash:
        # document doesn't exist, so add it
        print(f"new document {ref_doc_id}")
        self.docstore.set_document_hash(ref_doc_id, node.hash)
        deduped_nodes_to_run[ref_doc_id] = node
      elif existing_hash and existing_hash != node.hash:
        print(f"updated document {ref_doc_id}")
        self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)
        self.docstore.set_document_hash(ref_doc_id, node.hash)
        deduped_nodes_to_run[ref_doc_id] = node
        self.cleanup_fn(ref_doc_id)
      else:
        print(f"skipping document {ref_doc_id}")
        continue   # document exists and is unchanged, so skip it
    
    doc_ids_to_delete = existing_doc_ids_before - doc_ids_from_nodes
    for ref_doc_id in doc_ids_to_delete:
      print(f"deleting missing document {ref_doc_id}")
      self.docstore.delete_document(ref_doc_id)
      self.cleanup_fn(ref_doc_id)
    
    nodes_to_return = list(deduped_nodes_to_run.values())
    if len(nodes_to_return) > 0:
      self.cleanup_fn(ref_doc_id)


    return nodes_to_return