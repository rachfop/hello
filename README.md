```bash
uv venv
source .venv/bin/activate
# openai migrate
uv pip install -r requirements.txt
export OPENAI_API_KEY=<your-runpod-api-key>
python main.py
```