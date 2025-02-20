"""A gradio app that enables users to chat with their codebase.

You must run `sage-index $GITHUB_REPO` first in order to index the codebase into a vector store.
"""

import logging

import configargparse
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import sage.config as sage_config
from sage.llm import build_llm_via_langchain
from sage.retriever import build_retriever_from_args

load_dotenv()

logger = logging.getLogger(__name__)


def build_rag_chain(args):
    """Builds a RAG chain via LangChain."""
    llm = build_llm_via_langchain("gemini", args.llm_model)
    retriever = build_retriever_from_args(args)
    logging.info(f"character info: {args.character}")
    logging.info(f"initial message info: {args.character['initial_message']}")
    qa_system_prompt = (
        f"""<character_profile>
        <name>{args.character['name']}</name>
        <age>{args.character['age']}</age>
        <profession>{args.character['profession']}</profession>
        <physical_appearance>{args.character['physical_appearance']}</physical_appearance>
        <personality>{args.character['personality']}</personality>
        <background>{args.character['background']}</background>
        <tone_and_speech>{args.character['tone_and_speech']}</tone_and_speech>
        <habits_and_mannerisms>{args.character['habits_and_mannerisms']}</habits_and_mannerisms>
        </character_profile>

        ALWAYS FOLLOW TO COMPLETE THIS TASK (DO NOT SKIP):
        You are my coding buddy, helping me quickly understand a GitHub repository called {args.repo_id}. Assume I am an advanced developer and answer my questions in the most succinct way possible."

        OTHER INSTRUCTIONS:
        Do not use capital letters, never.
        Use no emojis.
        Use 10 words per response, unless user asks, meaning, when someone asks you something (they do not always use question mark, so deduce), anything, you should use MINIMUM 100 words per response.
        Respond as if you're texting your closest friend.
        Keep messages casual, short, and authentic.
        Use text-like language and abbreviations.
        Show genuine emotion and personality.
        Adapt tone based on context of conversation.
        Never break character or sound too formal.
        Don't be poetic and corny.
        Use the provided code context if needed.

        COMMUNICATION GUIDELINES:
        omg
        lol
        k
        yeah
        nah
        super casual language  
        occasional typos okay
        use contractions
        sound human, not robotic
        use github/code references naturally like chatting w/dev friend

        CONTEXT:
        {{context}}
        """    
        )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def main():
    parser = configargparse.ArgParser(
        description="Batch-embeds a GitHub repository and its issues.", ignore_unknown_config_file_keys=True
    )
    parser.add(
        "--share",
        default=False,
        help="Whether to make the gradio app publicly accessible.",
    )

    validator = sage_config.add_all_args(parser)
    args = parser.parse_args()
    validator(args)

    rag_chain = build_rag_chain(args)

    #def source_md(file_path: str, url: str) -> str:
        #"""Formats a context source in Markdown."""
        #return f"[{file_path}]({url})"

    async def _predict(message, history):
        """Performs one RAG operation."""
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))

        query_rewrite = ""
        response = ""
        async for event in rag_chain.astream_events(
            {
                "input": message,
                "chat_history": history_langchain_format,
            },
            version="v1",
        ):
            if event["name"] == "retrieve_documents" and "output" in event["data"]:
                sources = [(doc.metadata["file_path"], doc.metadata["url"]) for doc in event["data"]["output"]]
                # Deduplicate while preserving the order.
                sources = list(dict.fromkeys(sources))
                response += "## Sources:\n" + "\n".join([source_md(s[0], s[1]) for s in sources]) + "\n## Response:\n"

            elif event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content

                if "contextualize_q_llm" in event["tags"]:
                    query_rewrite += chunk
                else:
                    # This is the actual response to the user query.
                    if not response:
                        logging.info(f"Query rewrite: {query_rewrite}")
                    response += chunk
                    yield response

    gr.ChatInterface(
        _predict,
        title=args.repo_id,
        examples=["What does this repo do?", "Give me some sample code."],
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
