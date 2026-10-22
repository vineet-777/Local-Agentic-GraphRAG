"""
Local GraphRAG — FastAPI Backend
Serves the LangGraph engine + dynamic file upload/ingestion. Fully local.
"""

import asyncio
import json
import queue
import shutil
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

# Ensure project root is on sys.path so `core.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    is_lm_studio_running,
    is_db_ready,
    processing_status,
    UPLOADS_DIR,
    SYNTHESISER_MODEL,
    EMBEDDING_MODEL,
    ROUTER_MODEL,
)
from core.engine import run_query, generate_suggested_inquiries
from core.ingest import extract_text, chunk_documents, embed_chunks, ingest_files
from core.graph_builder import build_graph_from_chunks
from api.auth import (
    UserAlreadyExistsError,
    authenticate_user,
    create_session,
    create_user,
    get_current_user,
    initialize_auth_store,
    revoke_session,
)
from api.chats import (
    append_message,
    create_conversation,
    delete_conversation,
    get_conversation_with_messages,
    initialize_chat_store,
    list_conversations,
    rename_conversation,
)

_bearer_scheme = HTTPBearer(auto_error=False)


# ─── Background Processing ───────────────────────────────────────────────────

def _process_uploaded_files(file_paths: list[Path]):
    """
    Background task: ingest files into ChromaDB + build knowledge graph.
    Runs in a separate thread so the API stays responsive.
    """
    try:
        processing_status.start(total_files=len(file_paths))
        all_chunks_by_file = {}

        for i, file_path in enumerate(file_paths):
            file_name = file_path.name
            processing_status.update(
                current_file=file_name,
                processed_files=i,
                progress=(i / len(file_paths)) * 0.5,  # First 50% = extraction + embedding
            )

            # --- Stage 1: Extract & Chunk ---
            processing_status.update(
                stage="extracting",
                message=f"📄 Extracting text from {file_name}...",
            )
            documents = extract_text(file_path)

            if not documents:
                processing_status.update(
                    message=f"⚠️ No text found in {file_name}, skipping...",
                )
                continue

            processing_status.update(
                stage="chunking",
                message=f"✂️ Chunking {file_name} ({len(documents)} pages)...",
            )
            chunks = chunk_documents(documents)
            all_chunks_by_file[file_name] = chunks

            # --- Stage 2: Embed into ChromaDB ---
            processing_status.update(
                stage="embedding",
                message=f"🧮 Embedding {len(chunks)} chunks from {file_name}...",
                progress=((i + 0.5) / len(file_paths)) * 0.5,
            )
            new_count = embed_chunks(chunks)

            processing_status.update(
                message=f"✅ Embedded {new_count} new chunks from {file_name}",
            )

        # --- Stage 3: Build knowledge graph for all files ---
        total_files = len(all_chunks_by_file)
        for j, (file_name, chunks) in enumerate(all_chunks_by_file.items()):
            processing_status.update(
                stage="graphing",
                current_file=file_name,
                progress=0.5 + ((j / max(total_files, 1)) * 0.5),  # Second 50% = graph building
                message=f"🔗 Building knowledge graph from {file_name}...",
            )
            build_graph_from_chunks(chunks, file_name)

        processing_status.finish()

    except Exception as e:
        processing_status.fail(str(e))

    finally:
        # Clean up uploaded files
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    initialize_auth_store()
    initialize_chat_store()
    lm_status = is_lm_studio_running()
    db_status = is_db_ready()
    print("=" * 60)
    print("  Local GraphRAG — API Server Starting")
    print("=" * 60)
    print(f"  LM Studio:   {'✅ Connected' if lm_status.get('running') else '❌ ' + lm_status.get('error', 'Unavailable')}")
    print(f"  Router:      {'✅ ' + ROUTER_MODEL if lm_status.get('router_ready') else '❌ Model not loaded'}")
    print(f"  Synthesiser: {'✅ ' + SYNTHESISER_MODEL if lm_status.get('llm_ready') else '❌ Model not loaded'}")
    print(f"  Embeddings:  {'✅ ' + EMBEDDING_MODEL if lm_status.get('embed_ready') else '❌ Model not loaded'}")
    print(f"  ChromaDB:    {'✅ Found' if db_status.get('chromadb_exists') else '📭 Empty (upload files to start)'}")
    print(f"  Graph:       {'✅ ' + str(db_status.get('graph_nodes', 0)) + ' nodes' if db_status.get('graph_json_exists') else '📭 Empty'}")
    print("=" * 60)
    yield
    print("\n🛑 Local GraphRAG API shutting down.")


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Local GraphRAG API",
    description="Local GraphRAG API with dynamic file ingestion, powered by LM Studio",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    conversation_id: str | None = None


class RenameConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

class SigninRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    name: str
    created_at: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class QueryResponse(BaseModel):
    response: str
    route: str
    reasoning_trace: list[dict]


class SuggestedInquiriesRequest(BaseModel):
    conversation_id: str | None = None
    messages: list[dict] | None = None
    force_refresh: bool = False


class SuggestedInquiriesResponse(BaseModel):
    suggestions: list[str]
    context_type: str


def _sse_event(name: str, data: dict | str) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_query_response(
    query: str,
    request: Request,
    user_id: int,
    conversation_id: str,
    title: str,
) -> AsyncIterator[str]:
    events: queue.Queue[tuple[str, str | dict]] = queue.Queue()
    cancelled = threading.Event()

    def send_chunk(text: str) -> bool:
        if cancelled.is_set():
            return True
        events.put(("token", text))
        return cancelled.is_set()

    def execute_query() -> None:
        try:
            result = run_query(
                query,
                on_chunk=send_chunk,
                cancel_event=cancelled,
            )
        except Exception as exc:
            events.put(("error", f"Engine error: {exc}"))
        else:
            events.put(("result", result))

    worker = threading.Thread(target=execute_query, daemon=True)
    worker.start()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                name, data = await asyncio.to_thread(events.get, True, 0.1)
            except queue.Empty:
                continue
            if name == "token":
                yield _sse_event(name, {"text": data})
            elif name == "result":
                if await request.is_disconnected() or cancelled.is_set():
                    break
                if not isinstance(data, dict):
                    yield _sse_event("error", "Engine returned an invalid result.")
                    break
                result = data
                answer = result.get("response", "")
                trace = result.get("reasoning_trace", [])
                append_message(user_id, conversation_id, "assistant", answer, trace)
                yield _sse_event("complete", {
                    "route": result.get("route", "unknown"),
                    "reasoning_trace": trace,
                    "conversation_id": conversation_id,
                    "title": title,
                })
                break
            else:
                yield _sse_event(name, data)
                break
    finally:
        cancelled.set()
        await asyncio.to_thread(worker.join, 1)


class HealthResponse(BaseModel):
    status: str
    lm_studio: dict
    database: dict

class UploadResponse(BaseModel):
    message: str
    files_accepted: list[str]
    files_rejected: list[dict]

class ProcessingStatusResponse(BaseModel):
    is_processing: bool
    stage: str
    progress: float
    current_file: str
    total_files: int
    processed_files: int
    message: str
    error: str | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/conversations", tags=["Conversations"])
async def get_conversations(user: dict = Depends(get_current_user)):
    """List the signed-in user's conversations."""
    return list_conversations(user["id"])


@app.get("/api/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(conversation_id: str, user: dict = Depends(get_current_user)):
    """Load a conversation and its messages when owned by the signed-in user."""
    conversation = get_conversation_with_messages(user["id"], conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.patch("/api/conversations/{conversation_id}", tags=["Conversations"])
async def patch_conversation(
    conversation_id: str,
    request: RenameConversationRequest,
    user: dict = Depends(get_current_user),
):
    """Rename one of the signed-in user's conversations."""
    if not request.title:
        raise HTTPException(status_code=422, detail="Title must not be blank.")
    conversation = rename_conversation(user["id"], conversation_id, request.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.delete("/api/conversations/{conversation_id}", status_code=204, tags=["Conversations"])
async def remove_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete one of the signed-in user's conversations."""
    if not delete_conversation(user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")


@app.post("/api/auth/signup", response_model=AuthResponse, status_code=201, tags=["Authentication"])
async def signup(request: SignupRequest):
    """Create a local account and immediately start an authenticated session."""
    try:
        user = create_user(request.email, request.name, request.password)
    except UserAlreadyExistsError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    token, expires_in = create_session(user["id"])
    return AuthResponse(access_token=token, expires_in=expires_in, user=user)


@app.post("/api/auth/signin", response_model=AuthResponse, tags=["Authentication"])
async def signin(request: SigninRequest):
    """Verify credentials and issue a new bearer session."""
    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token, expires_in = create_session(user["id"])
    return AuthResponse(access_token=token, expires_in=expires_in, user=user)


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_signed_in_user(user: dict = Depends(get_current_user)):
    """Return the account associated with the supplied bearer session."""
    return user


@app.post("/api/auth/logout", status_code=204, tags=["Authentication"])
async def logout(
    _user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
):
    """Revoke the current bearer session."""
    if credentials is not None:
        revoke_session(credentials.credentials)

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health: LM Studio connectivity + database readiness."""
    lm_status = is_lm_studio_running()
    db_status = is_db_ready()

    all_ok = lm_status.get("running", False)

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        lm_studio=lm_status,
        database=db_status,
    )


@app.post("/api/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    _user: dict = Depends(get_current_user),
):
    """
    Upload PDF/DOCX files for ingestion.
    Files are saved to uploads/ and processed in the background.
    """
    # Check if already processing
    status = processing_status.get()
    if status["is_processing"]:
        raise HTTPException(
            status_code=409,
            detail="A file processing job is already running. Please wait for it to complete.",
        )

    # Check LM Studio
    lm_status = is_lm_studio_running()
    if not lm_status.get("running"):
        raise HTTPException(
            status_code=503,
            detail="LM Studio is not running. Please start LM Studio before uploading files.",
        )

    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

    accepted = []
    rejected = []
    saved_paths = []

    for file in files:
        ext = Path(file.filename).suffix.lower()

        # Validate extension
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append({"file": file.filename, "reason": f"Unsupported type: {ext}"})
            continue

        # Validate size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            rejected.append({"file": file.filename, "reason": "File exceeds 100MB limit"})
            continue

        # Save to uploads/
        save_path = UPLOADS_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(content)

        accepted.append(file.filename)
        saved_paths.append(save_path)

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail="No valid files to process.",
        )

    # Trigger background processing
    background_tasks.add_task(_process_uploaded_files, saved_paths)

    return UploadResponse(
        message=f"✅ {len(accepted)} file(s) accepted and queued for processing.",
        files_accepted=accepted,
        files_rejected=rejected,
    )


@app.get("/api/upload-status", response_model=ProcessingStatusResponse, tags=["Ingestion"])
async def get_upload_status(_user: dict = Depends(get_current_user)):
    """Poll this endpoint to track background file processing progress."""
    status = processing_status.get()
    return ProcessingStatusResponse(
        is_processing=status.get("is_processing", False),
        stage=status.get("stage", ""),
        progress=status.get("progress", 0.0),
        current_file=status.get("current_file", ""),
        total_files=status.get("total_files", 0),
        processed_files=status.get("processed_files", 0),
        message=status.get("message", ""),
        error=status.get("error"),
    )


@app.post(
    "/api/query",
    response_model=QueryResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
    tags=["Query"],
)
async def query_engine(
    request: QueryRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Main query endpoint. Runs the LangGraph pipeline and returns
    the answer + full reasoning trace for sidebar display.
    """
    # Pre-flight: LM Studio
    lm_status = is_lm_studio_running()
    if not lm_status.get("running"):
        raise HTTPException(
            status_code=503,
            detail="LM Studio is not running. Please start LM Studio and try again.",
        )

    # Pre-flight: DB (at least one source must exist)
    db_status = is_db_ready()
    if not db_status.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="No documents ingested yet. Please upload PDF/DOCX files first.",
        )

    if request.conversation_id is None:
        title_text = " ".join(request.query.split())
        title = title_text[:40].rstrip()
        if len(title_text) > 40:
            title = f"{title}…"
        conversation = create_conversation(user["id"], title)
        conversation_id = conversation["id"]
    else:
        conversation_id = request.conversation_id
        conversation = get_conversation_with_messages(user["id"], conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        title = conversation["title"]

    append_message(user["id"], conversation_id, "user", request.query)
    # TODO: pass history to run_query

    if "text/event-stream" in http_request.headers.get("accept", "").lower():
        return StreamingResponse(
            _stream_query_response(
                request.query, http_request, user["id"], conversation_id, title
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = run_query(request.query)
        trace = result.get("reasoning_trace", [])
        append_message(
            user["id"], conversation_id, "assistant",
            result.get("response", "No response generated."), trace,
        )
        return QueryResponse(
            response=result.get("response", "No response generated."),
            route=result.get("route", "unknown"),
            reasoning_trace=trace,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Engine error: {str(e)}",
        )


@app.post(
    "/api/suggested-inquiries",
    response_model=SuggestedInquiriesResponse,
    tags=["Query"],
)
async def get_suggested_inquiries(
    request: SuggestedInquiriesRequest,
    user: dict = Depends(get_current_user),
):
    """
    Generate dynamic, context-aware technical inquiries via LLM based on
    the knowledge base or current conversation.
    """
    conversation_history = request.messages or []
    if request.conversation_id and not conversation_history:
        conversation = get_conversation_with_messages(user["id"], request.conversation_id)
        if conversation:
            conversation_history = conversation.get("messages", [])

    result = generate_suggested_inquiries(
        conversation_history=conversation_history,
        num_suggestions=3,
    )
    return SuggestedInquiriesResponse(
        suggestions=result.get("suggestions", []),
        context_type=result.get("context_type", "knowledge_base"),
    )


@app.get("/", tags=["System"])
async def root():
    """Root info."""
    return {
        "message": "Local GraphRAG API v2.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /api/health",
            "upload": "POST /api/upload",
            "upload_status": "GET /api/upload-status",
            "query": "POST /api/query",
            "suggested_inquiries": "POST /api/suggested-inquiries",
        },
    }


# ─── Direct Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
