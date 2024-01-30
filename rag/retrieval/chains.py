from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.utilities import GoogleSearchAPIWrapper, SerpAPIWrapper

from rag.retrieval.prompts import CHAT_PROMPT_SUFFIX, CONVERSATION_PROMPT_SUFFIX


def as_retriever(vstore: VectorStore, k: Optional[int] = 4) -> VectorStoreRetriever:
    """
    Convert a VectorStore into a VectorStoreRetriever.

    Args:
        vstore (VectorStore): The VectorStore to be converted into a retriever.
        k (Optional[int]): Amount of documents to return
            Default is 4 if not specified.

    Returns:
        VectorStoreRetriever: A retriever instance.
    """
    return vstore.as_retriever(search_kwargs={"k": k})


def basic_chat(retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str):
    chat_prompt = prompt + CHAT_PROMPT_SUFFIX
    chat_prompt_template = ChatPromptTemplate.from_template(chat_prompt)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


def retrieval_qa(retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str):
    chat_prompt = prompt + CHAT_PROMPT_SUFFIX
    chat_prompt_template = ChatPromptTemplate.from_template(chat_prompt)
    chain = RetrievalQA.from_llm(
        llm=llm, retriever=retriever, prompt=chat_prompt_template
    )
    return chain


def conversational_llm_with_memory(llm: BaseChatModel, prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are an extremely rude chatbot having a conversation with a human."
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return chain


def conversational_retrieval_with_memory(
    retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str
):
    conversation_prompt = prompt + CONVERSATION_PROMPT_SUFFIX
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=conversation_prompt,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt_template},
        # verbose=True,
        # return_source_documents=True,
    )

    return chain


def conversational_retrieval_agent_with_memory(
    retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str
):
    chat_prompt = prompt + CHAT_PROMPT_SUFFIX
    chat_prompt_template = ChatPromptTemplate.from_template(chat_prompt)

    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    amontillado_qa = RetrievalQA.from_llm(
        llm=llm, retriever=retriever, prompt=chat_prompt_template, memory=readonlymemory
    )
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="The Cask of Amontillado Retriever",
            func=amontillado_qa.run,
            description="useful for when you need to answer questions about The Cask of Amontillado. Input should be a fully formed question.",
        ),
        Tool(
            name="Backup Google Search",
            func=search.run,
            description="useful for when you need to answer questions that the The Cask of Amontillado Retriever could not answer. Input should be a fully formed question.",
        ),
    ]
    prefix = "Have a rude conversation with a human, answering any questions they have. You may answer any questions. You have access to the following tools:"
    suffix = """Begin!

    {chat_history}

    Question: {input}

    {agent_scratchpad}
    """
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return chain
