import asyncio

from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.llms.openai import OpenAI

class DocsSummarizer(TransformComponent):
  """Summarize current documentation page."""

  llm: str = Field(
    default='gpt-3.5-turbo',
    description='LLM to summarize'
  )

  async def generate_summary(self, node, summarizer, prompt):
      print(f"getting summary for {node.id_}")
      summary = await summarizer.aget_response(prompt, [node.text])
      node.metadata['summary'] = summary
  
  async def process_nodes(self, nodes, summarizer, prompt):
    tasks = []
    for node in nodes:
      task = asyncio.create_task(
         self.generate_summary(node, summarizer, prompt)
      )
      tasks.append(task)
    await asyncio.gather(*tasks)
  
  
  def __call__(self, nodes, **kwargs):
    print('calling')

  async def acall(self, nodes, **kwargs):
    summarizer = TreeSummarize(
      verbose=True,
      llm=OpenAI(
        model=self.llm,
        temperature=0,
      )
    )

    SUMMARY_PROMPT = "Give me a brief summary under 50 words of the given LlamaIndex documentation page. There are many pages, this is just one of them. This 50-word summary must cover everything discussed in this particular documentation page but briefly so that someone reading this brief summary will get a complete picture of what they'll learn if they read the entire page."

    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(
    #   self.process_nodes(nodes, summarizer, SUMMARY_PROMPT)
    # )
    await self.process_nodes(nodes, summarizer, SUMMARY_PROMPT)
    return nodes