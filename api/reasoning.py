from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from api.detection import detect_json_internal
from api.human_detector import detect_human

router = APIRouter()

DANGEROUS_ANIMALS = ["lion", "tiger", "leopard", "jaguar", "elephant", "hyena"]

GREETINGS = [
    "hi", "hello", "hey", "how are you", "good morning", "good evening",
    "what's up", "sup", "good night", "good afternoon", "howdy",
    "what is your name", "what's your name", "who are you", "what are you",
    "tell me about yourself", "introduce yourself",
    "are you a bot", "are you ai", "are you real",
    "what can you do", "help me", "what do you do",
    "thanks", "thank you", "ok", "okay", "cool", "nice", "great",
    "bye", "goodbye", "see you", "later",
    "lol", "haha", "wow", "omg", "seriously",
]

GREETING_RESPONSES = {
    "what is your name": "I'm WildGuard — an AI-powered wildlife rescue and threat monitoring system.",
    "what's your name": "I'm WildGuard — an AI-powered wildlife rescue and threat monitoring system.",
    "who are you": "I'm WildGuard — an AI-powered wildlife rescue and threat monitoring system.",
    "what are you": "I'm WildGuard, a fine-tuned RT-DETR model that detects wildlife threats and triggers rescue alerts.",
    "what can you do": "I detect dangerous animals, identify human presence, and trigger rescue alerts when both are found in the same frame.",
    "help": "Upload an image and ask me: 'Is there a tiger?', 'Is it dangerous?', 'Is there a human?'",
    "introduce": "I'm WildGuard — built to detect wildlife intrusions and trigger rescue alerts in real time.",
    "are you a bot": "Yes — I'm WildGuard, an AI rescue alert system for wildlife monitoring.",
    "are you ai": "Yes — powered by RT-DETR for animal detection and MediaPipe for human detection.",
    "are you real": "Real enough to save lives! Ask me something about the image.",
    "thanks": "You're welcome! Ask me anything about the image.",
    "thank you": "Happy to help! Ask me anything about the image.",
    "bye": "Goodbye! Stay safe out there.",
    "goodbye": "Goodbye! Stay safe out there.",
    "see you": "See you! Come back anytime.",
    "later": "Later! Stay safe.",
    "lol": "Glad you're enjoying it! Ask me anything about the image.",
    "haha": "Ha! Now ask me something serious about the image.",
    "wow": "Right? Now ask me something about the image!",
    "omg": "I know! Ask me something about the image.",
    "seriously": "Dead serious. Ask me what's in the image!",
    "cool": "Thanks! Ask me anything about the image.",
    "nice": "Thanks! Ask me anything about the image.",
    "great": "Thanks! Ask me anything about the image.",
    "okay": "Sure! Ask me anything about the image.",
    "ok": "Sure! Ask me anything about the image.",
    "good morning": "Good morning! Ready to monitor wildlife?",
    "good evening": "Good evening! What would you like to know about this image?",
    "good night": "Good night! Come back anytime.",
    "good afternoon": "Good afternoon! Ask me anything about the image.",
    "howdy": "Howdy! Ask me anything about the wildlife in this image.",
    "hi": "Hello! I'm WildGuard. Ask me anything about the wildlife in this image.",
    "hello": "Hi there! I'm WildGuard. What would you like to know about the image?",
    "hey": "Hey! Ask me anything about the wildlife in this image.",
    "how are you": "Running at full capacity — detections live! Ask me something about the image.",
    "what's up": "Just monitoring wildlife! Ask me something about the image.",
    "sup": "Detecting threats! Ask me something about the image.",
}


def resolve_greeting(question: str) -> str:
    q = question.lower().strip()

    for key, response in GREETING_RESPONSES.items():
        if key in q:
            return response

    return "I'm WildGuard — your wildlife rescue assistant. Ask me anything about the image!"


DETECTION_KEYWORDS = [
    "how many", "count", "number of", "total", "how much",
    "is there", "are there", "any", "do you see", "can you see",
    "is a", "is an", "exists", "present", "spotted",
    "what animal", "which animal", "what is", "what are",
    "identify", "detect", "recognize", "classify", "tell me",
    "what do you see", "what can you see", "describe",
    "dangerous", "threat", "safe", "risk", "alert", "rescue",
    "warning", "attack", "predator", "prey", "hostile",
    "aggressive", "lurking", "hiding", "stalking",
    "lion", "tiger", "leopard", "jaguar", "elephant",
    "hyena", "buffalo", "baboon", "chimp", "vehicle",
    "animal", "mammal", "wildlife", "species", "cat",
    "where", "location", "position", "near", "close",
    "behind", "front", "left", "right", "corner",
    "moving", "running", "standing", "sitting", "eating",
    "drinking", "resting", "group", "herd", "pack", "alone",
    "night", "day", "dark", "bright", "camera trap",
    "poaching", "human", "person", "people", "intruder",
]


GUARDRAIL_KEYWORDS = [
    "weather", "temperature", "rain", "sunny", "cloudy",
    "wind", "humidity", "forecast", "climate", "hot", "cold",
    "time", "date", "year", "month", "when was", "what time",
    "hour", "minute", "second", "today", "yesterday",
    "sound", "noise", "smell", "odor", "hear", "listen",
    "mood", "feel", "emotion", "happy", "sad", "angry",
    "beautiful", "ugly", "aesthetic",
    "gps", "coordinates", "location name", "country",
    "camera model", "photographer", "who took",
    "resolution", "megapixel", "file size",
]


def route_visual_query(question: str) -> bool:
    q = question.lower().strip()

    for g in GREETINGS:
        if q == g or q.startswith(g):
            return False

    for keyword in GUARDRAIL_KEYWORDS:
        if keyword in q:
            return False

    for keyword in DETECTION_KEYWORDS:
        if keyword in q:
            return True

    return True


def build_threat_status(detections: list, human_result: dict) -> dict:
    human_detected = human_result["human_detected"]

    classes = [d["class"] for d in detections]

    counts = {}

    for c in classes:
        counts[c] = counts.get(c, 0) + 1

    dangerous_found = [
        c for c in counts
        if c in DANGEROUS_ANIMALS
    ]

    if dangerous_found and human_detected:
        return {
            "level": "RESCUE",
            "message": (
                f"🔴 RESCUE ALERT — {', '.join(dangerous_found)} "
                f"detected near a human. Immediate evacuation and "
                f"ranger dispatch required."
            ),
        }

    elif dangerous_found and not human_detected:
        return {
            "level": "BOUNDARY",
            "message": (
                f"🟡 BOUNDARY ALERT — {', '.join(dangerous_found)} "
                f"detected near reserve boundary. Ranger dispatch recommended."
            ),
        }

    elif human_detected and not dangerous_found:
        return {
            "level": "HUMAN",
            "message": (
                "🟢 Human presence detected. "
                "No dangerous wildlife in frame — monitor situation."
            ),
        }

    else:
        return {
            "level": "CLEAR",
            "message": "✅ No immediate threat detected.",
        }


def compose_visual_answer(
    question: str,
    detections: list,
    human_result: dict
) -> str:

    q = question.lower().strip()

    # ============================================================
    # 1. GREETING
    # ============================================================
    for g in GREETINGS:
        if q == g or q.startswith(g):
            return resolve_greeting(question)

    human_detected = human_result["human_detected"]

    classes = [d["class"] for d in detections]

    counts = {}

    for c in classes:
        counts[c] = counts.get(c, 0) + 1

    # ============================================================
    # 2. NO DETECTIONS
    # ============================================================
    if not detections and not human_detected:
        return (
            "No animals or humans were detected in this image "
            "with sufficient confidence."
        )

    # ============================================================
    # SUMMARY
    # ============================================================
    summary_parts = [
        f"{v} {k}(s)"
        for k, v in counts.items()
    ]

    if human_detected:
        summary_parts.append("1 human")

    summary = ", ".join(summary_parts)

    # ============================================================
    # 3. CAPABILITY CHECK
    #
    # These questions ask for information that RT-DETR /
    # object detection cannot determine from class, bbox,
    # and confidence alone.
    #
    # Examples:
    # "Is the tiger male?"
    # "Is the elephant injured?"
    # "Is the animal healthy?"
    # ============================================================
    if any(k in q for k in [
        "male",
        "female",
        "gender",
        "sex",

        "injured",
        "injury",
        "hurt",
        "wounded",

        "healthy",
        "health",
        "sick",
        "ill",
        "disease",

        "good",
        "bad",
        "okay",
        "ok",
        "fine",

        "happy",
        "sad",
        "angry",
        "calm",
        "afraid",
        "scared",

        "old",
        "young",
        "age",

        "pregnant",
        "pregnancy",

        "hungry",
        "thirsty",

        "sleeping",
        "dead",
        "alive"
    ]):
        return (
            "Insufficient information to determine that from "
            "the available detection results."
        )

    # ============================================================
    # 4. COUNT
    #
    # This MUST come before human / identification / fallback.
    #
    # Examples:
    # "How many are there?"
    # "How many elephants are there?"
    # "Count the tigers."
    # ============================================================
    if any(k in q for k in [
        "how many",
        "count",
        "number of",
        "total"
    ]):

        # Check if a specific detected class is mentioned
        matched_class = None

        for cls in counts:

            cls_lower = cls.lower()

            if (
                cls_lower in q
                or cls_lower.rstrip("s") in q
            ):
                matched_class = cls
                break

        # --------------------------------------------------------
        # Specific class count
        # --------------------------------------------------------
        if matched_class:

            count = counts[matched_class]

            if count == 1:
                return (
                    f"There is 1 {matched_class} "
                    f"in the image."
                )

            return (
                f"There are {count} {matched_class}s "
                f"in the image."
            )

        # --------------------------------------------------------
        # General count
        # --------------------------------------------------------
        total = len(detections) + (
            1 if human_detected else 0
        )

        return (
            f"There are {total} subjects in the image."
        )

    # ============================================================
    # 5. RESCUE / THREAT
    # ============================================================
    if any(k in q for k in [
        "dangerous",
        "threat",
        "safe",
        "risk",
        "alert",
        "rescue",
        "predator",
        "aggressive",
        "attack"
    ]):

        alert = build_threat_status(
            detections,
            human_result
        )

        return alert["message"]

    # ============================================================
    # 6. HUMAN PRESENCE
    # ============================================================
    if any(k in q for k in [
        "human",
        "person",
        "people",
        "man",
        "woman",
        "intruder"
    ]):

        if human_detected:

            dangerous_found = [
                c for c in counts
                if c in DANGEROUS_ANIMALS
            ]

            if dangerous_found:
                return (
                    f"🔴 RESCUE ALERT — Human detected alongside "
                    f"{', '.join(dangerous_found)}. "
                    f"Immediate action required."
                )

            return (
                "🟢 Human detected in this frame. "
                "No dangerous wildlife present."
            )

        return "No human detected in this frame."

    # ============================================================
    # 7. IDENTIFICATION
    # ============================================================
    if any(k in q for k in [
        "what is",
        "what are",
        "what animal",
        "which animal",
        "identify",
        "tell me",
        "describe"
    ]):

        return f"Detected in this frame: {summary}."

    # ============================================================
    # 8. PRESENCE
    # ============================================================
    if any(k in q for k in [
        "is there",
        "are there",
        "any",
        "do you see",
        "can you see",
        "exists",
        "present",
        "is that",
        "is this"
    ]):

        matched = [
            cls
            for cls in counts
            if cls.lower() in q
        ]

        if matched:

            cls_name = matched[0]

            conf = max(
                d["confidence"]
                for d in detections
                if d["class"] == cls_name
            )

            return (
                f"Yes — a {cls_name} was detected "
                f"with {conf:.0%} confidence."
            )

        if (
            "human" in q
            or "person" in q
            or "people" in q
        ):
            return (
                "Yes — a human was detected."
                if human_detected
                else
                "No human detected in this frame."
            )

        return (
            f"No — nothing matching your query was detected. "
            f"Found: {summary}."
        )

    # ============================================================
    # 9. VEHICLE
    # ============================================================
    if any(k in q for k in [
        "vehicle",
        "car",
        "truck",
        "poaching"
    ]):

        if "vehicle" in counts:
            return (
                "A vehicle was detected. "
                "Possible poaching activity — "
                "escalate to ranger station."
            )

        return "No vehicle detected in this image."

    # ============================================================
    # 10. POSITION
    # ============================================================
    if any(k in q for k in [
        "where",
        "location",
        "position"
    ]):

        if detections:

            d = detections[0]

            bbox = d["bbox"]

            return (
                f"The {d['class']} is at coordinates — "
                f"top-left ({bbox['x1']}, {bbox['y1']}), "
                f"bottom-right ({bbox['x2']}, {bbox['y2']})."
            )

        return "No detectable subject to locate."

    # ============================================================
    # 11. GROUP
    # ============================================================
    if any(k in q for k in [
        "group",
        "herd",
        "pack",
        "alone",
        "single"
    ]):

        total = len(detections) + (
            1 if human_detected else 0
        )

        if total == 1:

            subject = (
                classes[0]
                if classes
                else "human"
            )

            return (
                f"A single {subject} detected — "
                f"appears to be alone."
            )

        return (
            f"Multiple subjects detected: {summary}."
        )

    # ============================================================
    # 12. FALLBACK
    # ============================================================
    return f"Detected in this frame: {summary}."


@router.post("/")
async def process_wildlife_query(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    question = question.strip()

    q = question.lower().strip()

    # ============================================================
    # ROUTE QUESTION
    # ============================================================
    if not route_visual_query(question):

        return JSONResponse({
            "answer": (
                resolve_greeting(question)
                if any(
                    q == g or q.startswith(g)
                    for g in GREETINGS
                )
                else
                "Insufficient visual evidence to answer this question. "
                "This query requires information beyond what object "
                "detection can provide."
            ),

            "detection_used": False,

            "question": question
        })

    # ============================================================
    # READ IMAGE
    # ============================================================
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # ============================================================
    # RUN BOTH MODELS
    # ============================================================
    detections = await detect_json_internal(image)

    human_result = detect_human(image)

    # ============================================================
    # GENERATE ALERT
    # ============================================================
    alert = build_threat_status(
        detections,
        human_result
    )

    # ============================================================
    # GENERATE ANSWER
    # ============================================================
    answer = compose_visual_answer(
        question,
        detections,
        human_result
    )

    # ============================================================
    # API RESPONSE
    # ============================================================
    return JSONResponse({

        "answer": answer,

        "alert_level": alert["level"],

        "alert_message": alert["message"],

        "human_detected": human_result["human_detected"],

        "detection_used": True,

        "question": question,

        "detections": detections
    })