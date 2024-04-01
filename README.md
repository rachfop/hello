```bash
git clone https://github.com/rachfop/hello.git
cd hello
uv venv
source .venv/bin/activate
# openai migrate
uv pip install -r requirements.txt
export OPENAI_API_KEY=<your-runpod-api-key>
python3 main.py
```