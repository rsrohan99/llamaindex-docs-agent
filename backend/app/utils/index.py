import logging
import os
from pydantic import BaseModel
import shutil
from queue import Queue

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import FlatReader
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.indices import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.schema import NodeWithScore

from app.utils.node_parsers.markdown import CustomMarkdownNodeParser
from app.utils.transformations import URLExtractor, Deduplicator, Upserter
from app.utils.transformations import HyperlinksRemover, DocsSummarizer
from app.utils.misc import get_max_h_value

from typing import Optional, Dict, Any, List, Tuple

import os


PIPELINE_STORAGE_DIR = "./pipeline_storage"  # directory to cache the ingestion pipeline 
STORAGE_DIR = "./storage"
DATA_DIR = "./data"  # directory containing the documents to index


class EventObject(BaseModel):
    """
    Represents an event from the LlamaIndex callback handler.

    Attributes:
        type (str): The type of the event, e.g. "function_call".
        payload (dict): The payload associated with the event.
    """

    type: str
    payload: dict


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler specifically designed to stream function calls to a queue."""

    def __init__(self, queue: Queue) -> None:
        """Initialize the base callback handler."""
        super().__init__([], [])
        self._queue = queue
        self._counter = 0

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        if event_type == CBEventType.FUNCTION_CALL:
            self._queue.put(
                EventObject(
                    type="function_call",
                    payload={
                        "arguments_str": payload["function_call"],
                        "tool_str": payload["tool"].name,
                    },
                )
            )

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        # print(event_type)
        # print(payload)
        """Run when an event ends."""
        # if event_type == CBEventType.FUNCTION_CALL:
        #     # print(payload)
        #     self._queue.put(
        #         EventObject(
        #             type="function_call_response",
        #             payload={"response": payload["function_call_response"]},
        #         )
        #     )
        if event_type == CBEventType.AGENT_STEP:
            # put LLM response into queue
            self._queue.put(payload["response"])
        elif event_type == CBEventType.RETRIEVE:
            nodes_with_scores: list[NodeWithScore] = payload[EventPayload.NODES]
            nodes_to_return = []
            for node_with_score in nodes_with_scores:
                node = node_with_score.node
                node_meta = node.metadata
                # print(node_meta)
                if 'section_link' in node_meta:
                    nodes_to_return.append({
                        "id": node.id_,
                        "title": get_max_h_value(node_meta)
                                    or node_meta['file_path'],
                        "url": node.metadata['file_path'],
                        "section": node.metadata['section_link'],
                        "summary": node.metadata['summary'],
                    })
            # print(nodes_to_return)
            self._queue.put(
                EventObject(
                    type="nodes_retrieved",
                    payload={
                        "nodes": nodes_to_return
                    }
                )
            )

    @property
    def queue(self) -> Queue:
        """Get the queue of events."""
        return self._queue

    @property
    def counter(self) -> int:
        """Get the counter."""
        return self._counter

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass


def clean_old_persisted_indices(doc_id:str):
    dir_to_delete = f"{STORAGE_DIR}"
    if os.path.exists(dir_to_delete):
        print(f"Deleting dir: {dir_to_delete}")
        shutil.rmtree(dir_to_delete)

async def ingest(directory:str, docstore: BaseDocumentStore)->TextNode:
  reader = SimpleDirectoryReader(
    input_dir=directory,
    recursive=True,
    required_exts=[".md", ".mdx"],
    file_extractor={
      ".md": FlatReader(),
      ".mdx": FlatReader()
    }
  )

  docs = reader.load_data()

  deduplicator = Deduplicator(
    cleanup_fn=clean_old_persisted_indices,
    docstore=docstore,
  )
  url_extractor = URLExtractor(data_path=DATA_DIR)
  hyperlinks_remover = HyperlinksRemover()
  summarizer = DocsSummarizer(
    llm="gpt-3.5-turbo-0125"
  )
  upserter = Upserter(
    docstore=docstore,
    persist_dir=PIPELINE_STORAGE_DIR
  )

  pipeline = IngestionPipeline(
    transformations=[
        deduplicator,
        hyperlinks_remover,
        summarizer,
        url_extractor,
        upserter
    ],
)

  if os.path.exists(PIPELINE_STORAGE_DIR):
    pipeline.load(PIPELINE_STORAGE_DIR)
    nodes = await pipeline.arun(documents=docs)
  else:
    nodes = await pipeline.arun(documents=docs)
  pipeline.persist(PIPELINE_STORAGE_DIR)

  print(f"new nodes: {len(nodes)}")
  if len(nodes) > 0:
    # print(nodes)
    docstore.add_documents(nodes)
    docstore.persist(
      persist_path=os.path.join(PIPELINE_STORAGE_DIR, "docstore.json")
    )
    # if os.path.exists(STORAGE_DIR):
    #     print(f"Deleting {STORAGE_DIR}")
    #     shutil.rmtree(STORAGE_DIR)
  all_doc_ids = docstore.get_all_document_hashes().values()
#   print(all_doc_ids)
  return [docstore.get_document(doc_id=id) for id in all_doc_ids]


def _build_document_agents(
    storage_dir: str, docs: list[TextNode], callback_manager: CallbackManager
) -> Dict:
    """Build document agents."""
    node_parser = CustomMarkdownNodeParser()
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0125")
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    # Settings.callback_manager = callback_manager
    # service_context = ServiceContext.from_defaults(llm=llm)

    # Build agents dictionary
    agents = {}

    # this is for the baseline
    all_nodes = []

    for idx, doc in enumerate(docs):
        nodes = node_parser.get_nodes_from_documents([doc])
        all_nodes.extend(nodes)

        if not os.path.exists(f"./{storage_dir}/{doc.id_}"):
            # build vector index
            vector_index = VectorStoreIndex(
                nodes,
            )
            vector_index.storage_context.persist(
                persist_dir=f"./{storage_dir}/{doc.id_}"
            )
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(
                    persist_dir=f"./{storage_dir}/{doc.id_}"
                ),
            )

        # define query engines
        vector_index._callback_manager = callback_manager
        vector_query_engine = vector_index.as_query_engine()
        agents[doc.id_] = vector_query_engine

    return agents


def _build_top_agent(
    storage_dir: str, doc_agents: Dict, docstore: BaseDocumentStore,
    callback_manager: CallbackManager
) -> OpenAIAgent:
    """Build top-level agent."""
    # define tool for each document agent
    all_tools = []
    for doc_id in doc_agents.keys():
        doc = docstore.get_document(doc_id=doc_id)
        assert doc is not None
        wiki_summary = (
            f"This is the brief summary of one LlamaIndex documentation page: \"{doc.metadata['summary']}\". Use"
            f" this tool if you want to answer any questions about topics from the above summary. Please ask this tool a clearly specified elaborate query while using it and fully utilize it's response to answer the final query. Example input to this tool -> input: \"install llamaindex using node js?\". Don't use 1-2 word queries\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=doc_agents[doc_id],
            metadata=ToolMetadata(
                name=f"page_{doc.metadata['filename'].split('.')[0]}",
                description=wiki_summary,
            ),
        )
        all_tools.append(doc_tool)
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    # if obj_index doesn't already exist
    if not os.path.exists(f"./{storage_dir}/top"):
        storage_context = StorageContext.from_defaults()
        obj_index = ObjectIndex.from_objects(
            all_tools, tool_mapping, VectorStoreIndex, storage_context=storage_context
        )
        storage_context.persist(persist_dir=f"./{storage_dir}/top")
        # TODO: don't access private property

    else:
        # initialize storage context from existing storage
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./{storage_dir}/top"
        )
        index = load_index_from_storage(storage_context)
        obj_index = ObjectIndex(index, tool_mapping)

    top_agent = OpenAIAgent.from_tools(
        tool_retriever=obj_index.as_retriever(similarity_top_k=7),
        system_prompt=""" \
    You are an agent designed to answer queries about a Generative AI framework, LlamaIndex.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge. Pass the provided tools with clear and elaborate queries (e.g. "install llamaindex using node js?") and then fully utilize their response to answer the original query. When using multiple tools, break the original query into multiple elaborate queries and pass them to the respective tool as input. Be sure to make the inputs long and elaborate to capture entire context of the query. Don't use 1-2 word queries.\

    """,
        verbose=True,
        callback_manager=callback_manager,
    )

    return top_agent


async def get_agent():
    logger = logging.getLogger("uvicorn")

    if os.path.exists(PIPELINE_STORAGE_DIR):
        docstore = SimpleDocumentStore.from_persist_dir(PIPELINE_STORAGE_DIR)
    else:
        docstore = SimpleDocumentStore()

    docs = await ingest(
        # directory=f'{DATA_DIR}/docs/modules/ingestion_pipeline',
        # directory=f'{DATA_DIR}/docs/getting_started',
        directory=f'{DATA_DIR}/docs/modules',
        docstore=docstore,
    )

    # define callback manager with streaming
    queue = Queue()
    handler = StreamingCallbackHandler(queue)
    callback_manager = CallbackManager([handler])

    # build agent for each document
    doc_agents = _build_document_agents(
        STORAGE_DIR, docs, callback_manager=callback_manager
    )

    # build top-level agent
    top_agent = _build_top_agent(
        STORAGE_DIR, doc_agents, docstore, callback_manager
    )

    logger.info(f"Built agent.")

    return top_agent
