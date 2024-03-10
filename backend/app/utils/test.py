import os

from dotenv import load_dotenv

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import FlatReader
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.utils.node_parsers.markdown import CustomMarkdownNodeParser
from app.utils.transformations import URLExtractor, Deduplicator, Upserter
from app.utils.transformations import HyperlinksRemover, DocsSummarizer

load_dotenv()

DATA_PATH="./data"
PIPELINE_STORAGE_PATH="./pipeline_storage"
def main():
  reader = SimpleDirectoryReader(
    input_dir=f'{DATA_PATH}/docs/getting_started',
    required_exts=[".md", ".mdx"],
    file_extractor={
      ".md": FlatReader(),
      ".mdx": FlatReader()
    }
  )

  docs = reader.load_data()
  # for doc in docs:
  #   doc.metadata["url_path"], _ = splitext(relpath(
  #     doc.metadata['file_path'], DATA_PATH+"/docs"
  #   ))


  print(len(docs))
  # print(docs[1])
  if os.path.exists(PIPELINE_STORAGE_PATH):
    docstore = SimpleDocumentStore.from_persist_dir(PIPELINE_STORAGE_PATH)
  else:
    docstore = SimpleDocumentStore()

  deduplicator = Deduplicator(
    docstore=docstore,
  )
  hyperlinks_remover = HyperlinksRemover()
  url_extractor = URLExtractor(data_path=DATA_PATH)
  summarizer = DocsSummarizer(
    llm="gpt-3.5-turbo-0125"
  )
  upserter = Upserter(
    docstore=docstore,
    persist_dir=PIPELINE_STORAGE_PATH
  )
  node_parser = CustomMarkdownNodeParser()

  pipeline = IngestionPipeline(
    transformations=[
      deduplicator,
      hyperlinks_remover,
      summarizer,
      url_extractor,
      upserter
   ],
  )

  if os.path.exists(PIPELINE_STORAGE_PATH):
    pipeline.load(PIPELINE_STORAGE_PATH)
    pipeline.run(documents=docs)
  else:
    pipeline.run(documents=docs)
    pipeline.persist(PIPELINE_STORAGE_PATH)


  # nodes = node_parser.get_nodes_from_documents(docs)
  all_doc_ids = docstore.get_all_document_hashes().values()
  print(all_doc_ids)
  all_docs = [docstore.get_document(doc_id=id) for id in docstore.get_all_document_hashes().values()]
  # print(all_docs.get_content(metadata_mode=MetadataMode.ALL))

  print(len(all_docs))
  # print(nodes[2:6])
  # print(all_docs[2].get_content(metadata_mode=MetadataMode.ALL))
  print(all_docs[2].metadata)