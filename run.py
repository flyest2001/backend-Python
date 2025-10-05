# run.py
import uvicorn

if __name__ == "__main__":
    # Point uvicorn to the 'app' object inside the 'app.api' module
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)