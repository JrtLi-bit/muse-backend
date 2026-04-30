from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from PIL import Image
import io
import base64
import traceback
import os  # <-- ADDED THIS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# <-- CHANGED THIS: Securely pulling the key from Render's hidden settings
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY environment variable is missing.")

client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-3.1-flash-lite'

class CanvasData(BaseModel):
    image_data: str
    is_final: bool
    red_prompt: str
    blue_prompt: str

@app.post("/analyze")
async def analyze_game(data: CanvasData):
    try:
        img_str = data.image_data.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))
        
        if not data.is_final:
            # The prompt for normal rounds
            prompt_text = f"""
            You are Muse, a sarcastic, terminally online commentator fluent in Gen-Alpha/gen-z slang.
            1. On the first line, pin a probability leaderboard (the top 3 objects this image looks like, with percentages). 
            2. Start a new paragraph and deliver an absurd roast (around 25 words) mocking the current progress of this patched-together mess. Keep the language accessible—sharp and snarky, but not overly offensive. So something like "Even I can draw better than this in procreate if I have hands.". If you mention mention MS paint, say Microsoft Paint. Also you can use some internet slang like "tbh, on god, ngl", or say like "this is like a drawing from kid who says six-seven all day",these type of words, not have to be exactly the same.
            """
        else:
            # The prompt for the final verdict
            prompt_text = f"""
            This is the final result of the showdown. Red Team's goal was to draw '{data.red_prompt}', and Blue Team's goal was '{data.blue_prompt}'.
            Please deliver the final verdict as the ultimate judge:
            1. On the first line, announce the winning team (based on whose target object is more visually recognizable in the drawing).
            2. Deliver a final roast (around 50 words) humorously critiquing and mocking their "art." Keep the language accessible—sharp and snarky, but not overly offensive.
            """
        
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=[prompt_text, img]
        )
        return {"reply": response.text}
        
    except Exception as e:
        print(traceback.format_exc())
        return {"reply": f"🚨 Backend error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # <-- CHANGED THIS: 0.0.0.0 allows Render to host it publicly
    uvicorn.run(app, host="0.0.0.0", port=8000)
