import os
from pydantic import BaseModel
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

from agents.conversation_llm import ConversationalAgent
from agents.segments import SegmentsAgent
from agents.dialogue_agent import DialogueAgent
from agents.nutrition_analysis import NutritionAgent
load_dotenv()

class UserQuery(BaseModel):
    img: str | None = None
    plate_diameter: float | None = None
    query: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # threading.Thread(target=run_pathway, daemon=True).start()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise("Error: API key not found in config.py")

    app.state.llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    app.state.tavily_client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    app.state.ConversationalAgent = ConversationalAgent(app.state.llm_client)
    app.state.SegmentationAgent = SegmentsAgent(app.state.http_client, app.state.llm_client, os.getenv("VOLUME_ESTIMATOR_ENDPOINT", 'http://192.168.1.191:5000/predict'))
    app.state.DialogueAgent = DialogueAgent(app.state.llm_client)
    app.state.NutritionAgent = NutritionAgent(app.state.tavily_client, app.state.llm_client)
    print("HTTP client opened.")
    yield
    await app.state.http_client.aclose()
    print("HTTP client closed.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/evaluate")
async def evaluate(websocket: WebSocket):
    await websocket.accept()

    try:
        # Receive initial request
        payload = UserQuery.model_validate(await websocket.receive_json())
        if not payload.img and not payload.query:
            # close current session and return missing values error
            await websocket.send_json({
                "error": "Either 'img' or 'query' must be provided."
            })
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Missing required values"
            )
        
        if payload.img:
            volumes = await app.state.SegmentationAgent(payload.img, format='base_64')
            if len(volumes.get('segments', [])) == 0:
                await websocket.send_json({
                    "error": "No segment detected"
                })
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Missing required values"
                )
                return
            need_justifications = app.state.DialogueAgent.confirm_analysis(volumes)
            if need_justifications:
                for justification in need_justifications:
                    await websocket.send_json({
                        "type": "question",
                        # 'segment_id': justification['segment_id'],
                        "question": justification['question_asked'],
                        'uncertainities': justification['major_uncertainties']
                    })
                    justification['justification'] = (await websocket.receive_json())['answer']
                await app.state.DialogueAgent._parse_bulk_answers(need_justifications, volumes)
            
            keys_to_delete = ['major_uncertainties', 'most_important_question', 'ambiguity_flag']
            for segment in volumes['segments']:
                for k in keys_to_delete:
                    segment.pop(k, None)  # None prevents KeyError
            
            answer = await app.state.NutritionAgent(volumes, payload.query, payload.img)
            await websocket.send_json(volumes)
            await websocket.send_json({
                "type": "result",
                # **volumes,
                "answer":answer
            })

        await websocket.close()
        # answer = await websocket.receive_json()

        # user_response = answer["answer"]

        # Continue evaluation...


    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)