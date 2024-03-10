from llama_index.core.vector_stores.types import BasePydanticVectorStore

class DummyVectorStore(BasePydanticVectorStore):
  stores_text = False
  def client(self):
    ...

  def add(self, _):
    ...
  
  def delete(self, _, **__):
    ...

  def query(self, _, **__):
    ...