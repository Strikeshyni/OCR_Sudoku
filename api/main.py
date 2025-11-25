from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil

app = FastAPI(title="OCR Sudoku API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
OUTPUT_IMAGE = "output_api.png"
DEBUG_IMAGES = [
    "debug_1_gray.png",
    "debug_2_blurred.png",
    "debug_3_binary.png",
    "debug_4_grid_detected.png",
    "debug_5_rectified.png",
    "debug_6_cells.png"
]

@app.post("/solve")
async def solve_sudoku(file: UploadFile = File(...)):
    # Ensure data directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    file_location = os.path.join(UPLOAD_DIR, "api_upload.png")
    
    # Save uploaded file
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run C program
    # Assuming running from root: ./build/sudoku_solver
    # Usage: ./build/sudoku_solver <input_image> <output_image>
    command = ["./build/sudoku_solver", file_location, OUTPUT_IMAGE]
    
    try:
        # Run with timeout to prevent infinite loops if they still exist
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
             raise HTTPException(status_code=500, detail=f"Solver failed: {result.stderr}")
             
        return {
            "message": "Sudoku processed successfully",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Solver timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running solver: {str(e)}")

@app.get("/debug-images")
def list_debug_images():
    # Check which exist
    existing = [img for img in DEBUG_IMAGES if os.path.exists(img)]
    return {"images": existing}

@app.get("/debug-images/{image_name}")
def get_debug_image(image_name: str):
    # Allow getting debug images and the output image
    allowed = DEBUG_IMAGES + [OUTPUT_IMAGE]
    
    if image_name not in allowed:
        raise HTTPException(status_code=404, detail="Image not found or not allowed")
    
    if not os.path.exists(image_name):
        raise HTTPException(status_code=404, detail="Image file does not exist")
        
    return FileResponse(image_name)
