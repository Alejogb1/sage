import asyncio
import json
import logging
from typing import Optional
import time

import configargparse
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import socketio

import sage.config as sage_config
from sage.chat import build_rag_chain
from langchain.schema import AIMessage, HumanMessage
chat_histories = {}
# Create a Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
socket_app = socketio.ASGIApp(sio)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"]   # Allows all headers
)

# Mount socket.io app
app.mount("/socket.io", socket_app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def source_md(file_path: str, url: str) -> str:
    """Formats a context source in Markdown."""
    return f"[{file_path}]({url})"

@sio.on('connect')
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.on('disconnect')
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.on('message')
@sio.on('message')
@sio.on('message')
async def handle_message(sid, data):
    try:
        message_data = data if isinstance(data, dict) else json.loads(data)
        logger.info(f"Received message: {message_data}")
        user_input = message_data.get('content', '')
        chat_id = message_data.get('chatId', '')
        
        # Extract repo_url from the client's message data
        repo_url = message_data.get('repoUrl', 'michaelshimeles/friend.com')  # Default fallback if not provided
        repo_url = f"{repo_url['owner']}/{repo_url['repo']}"
        
        logger.info(f"Repo URL: {repo_url}")
        logger.info(f"Processing input: {user_input}")

        # Initialize chat history for new sessions
        if sid not in chat_histories:
            chat_histories[sid] = []
        
        # Convert existing chat history to LangChain format
        history_langchain_format = []
        for msg in chat_histories[sid]:
            if isinstance(msg, HumanMessage):
                history_langchain_format.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_langchain_format.append({"type": "ai", "content": msg.content})

        # Add the new user message to the history
        chat_histories[sid].append(HumanMessage(content=user_input))
        history_langchain_format.append({"type": "human", "content": user_input})

        assistant_message = {
            'role': 'assistant',
            'content': '',
            'timestamp': int(time.time() * 1000),
            'chatId': chat_id
        }

        # Build RAG chain with repo_url from the client's message
        parser = configargparse.ArgParser()
        validator = sage_config.add_all_args(parser)
        parser.set_defaults(
            repo_id=repo_url,  # Use the repo_url from the client's message
            embedding_provider='gemini',
            llm_retriever=True,
            embedding_model='text-embedding-004',
            llm_provider='gemini',
            llm_model='gemini-2.0-flash-exp',
            vector_store_provider='pinecone',
            retrieval_alpha=0.5,
            retriever_top_k=5,
            character=message_data.get('character', '')
        )

        args = parser.parse_args()  # Empty list to use defaults
        validator(args)
        rag_chain = build_rag_chain(args)

        async for chunk in rag_chain.astream_events(
            {"input": user_input, "chat_history": history_langchain_format},
            version="v1",
        ):
        
            if chunk["event"] == "on_chat_model_stream":
                assistant_message['content'] += chunk["data"]["chunk"].content
                assistant_message['timestamp'] = int(time.time() * 1000)
        
        logger.info(f"Sending response: {assistant_message}")
        await sio.emit('message', assistant_message, room=sid)

        # Add the assistant's response to the history
        chat_histories[sid].append(AIMessage(content=assistant_message['content']))

        logger.info(f"Updated chat history length: {len(chat_histories[sid])}")

    except Exception as e:
        error_message = {
            'role': 'assistant',
            'content': f"Error: {str(e)}",
            'timestamp': int(time.time() * 1000),
            'chatId': chat_id
        }
        logger.error(f"Error processing message: {e}", exc_info=True)
        await sio.emit('message', error_message, room=sid)
async def run_server(host: str = "0.0.0.0", port: int = 3001):
    """Run the WebSocket server using uvicorn."""
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    import os
    asyncio.run(run_server())