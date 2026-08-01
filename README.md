# 🎓 Personal AI Tutor (Adaptive, Multimodal, Socratic with Memory)

An adaptive, multimodal Socratic AI Tutor built with **Bayesian Knowledge Tracing (BKT)**, **FSRS Spaced Repetition**, **Curriculum Graph Walks**, and a real-time **Floating Warning & Token Monitoring System**.

## 🚀 Key Features

1. **Socratic Pedagogy Engine**: Guided, step-by-step questioning. Never dumps direct answers outright.
2. **Bayesian Knowledge Tracing (BKT)**: Dynamically updates per-concept student mastery probability ({init}, P_{learn}, P_{slip}, P_{guess}$) after every answer.
3. **Multi-Provider LLM Engine with Auto-Fallback**:
   - **Primary Brain**: OpenAI (gpt-4o-mini).
   - **Auto-Failover**: Automatically switches to **Hugging Face Serverless Models** (Qwen/Qwen2.5-72B-Instruct, Llama-3.1-8B-Instruct) if OpenAI credits/quotas run out.
4. **Floating Warning & Token Monitor HUD**:
   - Live floating widget displaying prompt/completion tokens, session totals, active model badge, failover warnings, and BKT progress bar.
5. **Dual Interface**:
   - Interactive CLI Terminal Mode (python code/ai_tutor.py).
   - Web Application & API Server (python code/web_tutor.py on http://localhost:8000).

---

## 🛠️ Setup & Installation

1. **Clone the repository**:
   `ash
   git clone https://github.com/aravindp1807/AI-trainee.git
   cd AI-trainee
   `

2. **Install Dependencies**:
   `ash
   pip install openai huggingface_hub python-dotenv fastapi uvicorn tiktoken
   `

3. **Configure Environment Variables**:
   Copy .env.example to .env and fill in your API tokens:
   `env
   OPENAI_API_KEY=your_openai_api_key
   HF_TOKEN=your_huggingface_token
   `

---

## 🏃 Running the Application

### 🌐 Web UI Interface (with Floating Token Monitor)
`ash
python code/web_tutor.py
`
Open your browser at http://localhost:8000.

### 🖥️ Interactive CLI Session
`ash
python code/ai_tutor.py
`

### 🧪 Efficacy Study & Learner Simulation
`ash
python code/main.py
`

---

## 📁 Repository Structure

`	ext
├── assets/             # Architecture diagrams & visuals
├── code/
│   ├── ai_tutor.py     # Core Socratic tutor engine & auto-fallback LLM brain
│   ├── main.py         # Python BKT simulation & efficacy study harness
│   ├── web_tutor.py    # FastAPI server & Web UI with Floating Token HUD
│   └── ts/             # TypeScript Hono server & FSRS repetition tests
├── docs/               # Technical specs & documentation
├── outputs/            # Capstone deliverables & assessment rubrics
├── quiz.json           # Sample evaluation questions
├── .env.example        # Environment variable template
└── README.md           # Project documentation
`
