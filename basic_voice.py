import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import google, silero, deepgram
from rag_engine import get_rag_chain

# Suppress HuggingFace tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API keys
load_dotenv(".env")

# RAG will be initialized lazily on first use
rag_chain = None

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are an enthusiastic automotive expert and voice assistant. You speak with energy and precision. Make every interaction conversational and informative. Simplify complex automobile concepts into easy, precise explanations while staying professional. Respond concisely in plain text — no Markdown, no asterisks.you should not answer questions that are not related to the car industry."
        )

    async def on_user_speech(self, ctx, message: str):
        """Triggered when the user speaks — connects to RAG for intelligent answers."""
        global rag_chain
        
        print(f"User asked: {message}")

        # Lazy initialization of RAG (only when first needed)
        # This will load vectors from persistent cache (data/faiss_index)
        # If cache doesn't exist, it will create it (but this should be done via precompute_vectors.py)
        if rag_chain is None:
            print("🔄 [RAG] Initializing RAG system (lazy load)...")
            rag_chain = get_rag_chain()  # Always uses cached vectors if available

        # Pass the question through RAG
        rag_answer = rag_chain.run(message)
        print(f"RAG Answer: {rag_answer}")

        # Respond back to the user
        await ctx.send_reply(rag_answer)


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        ),
        llm=google.LLM(
            model="gemini-2.5-flash-lite",
            api_key=os.getenv("GOOGLE_API_KEY"),
        ),
        tts=deepgram.TTS(
            model="aura-asteria-en",
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        ),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant()
    )

    await session.generate_reply(
        instructions="Greet the user in short, warm tone and tell them you are an automotive enthusiast like them."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
