from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.detection import router as detection_router, load_model
from api.reasoning import router as reasoning_router

app = FastAPI(title="WildGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    load_model("models/wildguard/weights/best.pt")

app.include_router(detection_router, prefix="/detect", tags=["Detection"])
app.include_router(reasoning_router, prefix="/analyze", tags=["Reasoning"])

app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def root():
    return FileResponse("api/static/index.html")