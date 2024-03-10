### Advanced chatbot over LlamaIndex TS documentation 🔥

https://github.com/rsrohan99/llamaindex-docs-agent/assets/62835870/42fbd1ba-c42f-4b86-b33e-093272d76639

# Multi-document Agents

This is a [LlamaIndex](https://www.llamaindex.ai/) project bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama).

This multi-document agent is built over the LlamaIndex.TS documentation.

We use our multi-document agent architecture:

- Individual query engine per document
- Top level Orchestrator agent across documents that can pick relevant subsets

This also streams _all_ intermediate results from the agent via a custom Callback handler.

We use this Custom Callback handler to also send intermediate nodes that are retrieved during retrieval of document level query engines, to the frontend.

It allows us to show the relevant section of the documentation in the preview window.

## Main Files to Look At

This extends beyond the simple `create-llama` example. To see changes, look at the following files:

- `backend/app/utils/index.py` - contains core logic for constructing + getting multi-doc agent
- `backend/app/api/routers/chat.py` - contains implementation of chat endpoint + threading to stream intermediate responses.

We also created some custom `Transformations` that we use with out robust `IngestionPipeline`

As we update the documentations in the `data` folder, this `IngestionPipeline` takes care of handling duplicates, applying our custom nodes transformation logic etc.

The custom transformations we've used:

- `Deduplicator` - handles duplicates.
- `HyperlinksRemover` - cleans the markdown files.
- `Summarizer` - creates summary of the node and adds that as a metadata.
- `URLExtractor` - generates the url of a particular node section.
- `Upserter` - updates the docstore with new and updated nodes, deletes old ones.

## Getting Started

First, startup the backend as described in the [backend README](./backend/README.md).

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex (Python features).
- [LlamaIndexTS Documentation](https://ts.llamaindex.ai) - learn about LlamaIndex (Typescript features).

You can check out [the LlamaIndexTS GitHub repository](https://github.com/run-llama/LlamaIndexTS) - your feedback and contributions are welcome!
