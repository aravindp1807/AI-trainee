import os
import sys
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tiktoken

load_dotenv()

@dataclass
class BKTParams:
    p_init: float = 0.20
    p_learn: float = 0.15
    p_slip: float = 0.10
    p_guess: float = 0.20

def bkt_update(mastery: float, correct: bool, params: BKTParams = BKTParams()) -> float:
    if correct:
        num = mastery * (1.0 - params.p_slip)
        denom = num + (1.0 - mastery) * params.p_guess
    else:
        num = mastery * params.p_slip
        denom = num + (1.0 - mastery) * (1.0 - params.p_guess)
    posterior = num / max(denom, 1e-6)
    return min(1.0, posterior + (1.0 - posterior) * params.p_learn)

@dataclass
class ConceptNode:
    id: str
    title: str
    prereqs: List[str]
    description: str

ALGEBRA_CURRICULUM = [
    ConceptNode('number_line', 'Number Line and Absolute Value', [], 'Understanding positive and negative positions on a number line.'),
    ConceptNode('addition_subtraction', 'Addition and Subtraction of Integers', ['number_line'], 'Combining signed numbers.'),
    ConceptNode('multiplication_division', 'Multiplication and Division of Integers', ['addition_subtraction'], 'Multiplying and dividing signed numbers.'),
    ConceptNode('equality', 'Equality and Balance Principle', ['addition_subtraction'], 'Whatever operation is applied to one side must be applied to the other.'),
    ConceptNode('isolating_variable_one_step', 'One-Step Linear Equations', ['equality', 'addition_subtraction'], 'Solving equations like x + 5 = 12.'),
    ConceptNode('isolating_variable_two_step', 'Two-Step Linear Equations', ['isolating_variable_one_step', 'multiplication_division'], 'Solving equations like 3x + 6 = 12.'),
    ConceptNode('distributive_property', 'Distributive Property', ['multiplication_division'], 'Expanding expressions like a(b + c) = ab + ac.'),
    ConceptNode('linear_equations', 'Multi-Step Linear Equations', ['isolating_variable_two_step', 'distributive_property'], 'Solving equations requiring multiple steps.'),
]

class TokenTracker:
    def __init__(self):
        self.encoder = tiktoken.get_encoding('cl100k_base')
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.warnings: List[str] = []
        self.openai_failed = False
        self.max_free_hf_tokens_per_min = 4000

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.encoder.encode(text))
        except Exception:
            return len(text) // 4

    def record_usage(self, prompt_text: str, completion_text: str, provider: str, failover_occurred: bool = False):
        p_tok = self.count_tokens(prompt_text)
        c_tok = self.count_tokens(completion_text)
        
        self.last_turn_prompt_tokens = p_tok
        self.last_turn_completion_tokens = c_tok
        self.session_prompt_tokens += p_tok
        self.session_completion_tokens += c_tok
        
        self.warnings.clear()
        
        if failover_occurred or 'Hugging Face' in provider:
            self.warnings.append('⚡ OpenAI quota exhausted! Automatically switched to Hugging Face free tier models.')
            
        total_session = self.session_prompt_tokens + self.session_completion_tokens
        if total_session > 10000:
            self.warnings.append(f'&#9266; High session token usage ({total_session:,} tokens consumed).')

    def get_stats(self, active_provider: str) -> Dict:
        total_session = self.session_prompt_tokens + self.session_completion_tokens
        status_level = 'NORMAL'
        if self.warnings:
            status_level = 'FAILOVER'
            
        return {
            'last_prompt_tokens': self.last_turn_prompt_tokens,
            'last_completion_tokens': self.last_turn_completion_tokens,
            'last_turn_total': self.last_turn_prompt_tokens + self.last_turn_completion_tokens,
            'session_prompt_tokens': self.session_prompt_tokens,
            'session_completion_tokens': self.session_completion_tokens,
            'session_total_tokens': total_session,
            'active_provider': active_provider,
            'status_level': status_level,
            'warnings': self.warnings,
            'estimated_remaining_hf_rate': max(0, self.max_free_hf_tokens_per_min - (self.last_turn_prompt_tokens + self.last_turn_completion_tokens))
        }

class LLMBrain:
    def __init__(self, tracker: TokenTracker):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')
        self.tracker = tracker
        self.hf_models = [
            'Qwen/Qwen2.5-72B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct',
            'Qwen/Qwen2.5-Coder-32B-Instruct'
        ]
        self.active_provider = 'Initializing...'

    def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        full_messages = [{'role': 'system', 'content': system_prompt}] + messages
        prompt_full_text = system_prompt + '\n' + '\n'.join([u['content'] for u in messages])
        
        failover = False
        
        # 1. Try OpenAI
        if self.openai_key and not self.tracker.openai_failed:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=full_messages,
                    temperature=0.7,
                    max_tokens=400
                )
                reply = response.choices[0].message.content.strip()
                self.active_provider = 'OpenAI (gpt-4o-mini)'
                self.tracker.record_usage(prompt_full_text, reply, self.active_provider)
                return reply
            except Exception as e:
                self.tracker.openai_failed = True
                failover = True

        # 2. Fallback to Hugging Face
        if self.hf_token:
            from huggingface_hub import InferenceClient
            client = InferenceClient(api_key=self.hf_token)
            for m in self.hf_models:
                try:
                    response = client.chat_completion(
                        model=m,
                        messages=full_messages,
                        temperature=0.7,
                        max_tokens=400
                    )
                    reply = response.choices[0].message.content.strip()
                    self.active_provider = 'Hugging Face (' + m + ')'
                    self.tracker.record_usage(prompt_full_text, reply, self.active_provider, failover_occurred=failover)
                    return reply
                except Exception as hf_err:
                    continue

        reply = 'AI Model temporarily unavailable. Please verify API keys.'
        self.active_provider = 'None (Offline)'
        self.tracker.record_usage(prompt_full_text, reply, self.active_provider, failover_occurred=True)
        return reply

class PersonalAITutor:
    def __init__(self):
        self.curriculum = {c.id: c for c in ALGEBRA_CURRICULUM}
        self.mastery: Dict[str, float] = {c.id: 0.20 for c in ALGEBRA_CURRICULUM}
        self.history: List[Dict] = []
        self.tracker = TokenTracker()
        self.brain = LLMBrain(self.tracker)
        self.current_concept_id: str = ALGEBRA_CURRICULUM[0].id

    def get_next_concept(self) -> Optional[ConceptNode]:
        for c in self.curriculum.values():
            if self.mastery[c.id] < 0.85:
                if all(self.mastery[p] >= 0.70 for p in c.prereqs):
                    return c
        return None

    def evaluate_response(self, learner_input: str, concept: ConceptNode) -> bool:
        eval_prompt = 'Concept: ' + concept.title + '. Student answer: ' + repr(learner_input) + '. Is this mathematically correct? Reply ONLY CORRECT or INCORRECT.'
        res = self.brain.generate([{'role': 'user', 'content': eval_prompt}], 'Evaluate math answers.')
        return 'CORRECT' in res.upper()

    def chat_turn(self, user_message: str) -> Tuple[str, Dict]:
        current_node = self.curriculum.get(self.current_concept_id, ALGEBRA_CURRICULUM[0])
        
        if user_message.strip():
            is_correct = self.evaluate_response(user_message, current_node)
            old_m = self.mastery[current_node.id]
            new_m = bkt_update(old_m, is_correct)
            self.mastery[current_node.id] = new_m
            
            if new_m >= 0.85:
                nxt = self.get_next_concept()
                if nxt and nxt.id != current_node.id:
                    self.current_concept_id = nxt.id
                    current_node = nxt

        mastery_pct = str(int(self.mastery[current_node.id] * 100))
        socratic_system_prompt = (
            'You are a K-12 Socratic AI Tutor teaching Algebra.\n'
            'Rules:\n'
            '1. NEVER give the direct final answer outright.\n'
            '2. Ask leading questions and guide step-by-step.\n'
            '3. If stuck, provide a smaller scaffolded hint.\n'
            '4. Current target topic: ' + current_node.title + ' (Mastery: ' + mastery_pct + '%).\n'
            'Overview: ' + current_node.description
        )
        
        messages = []
        for h in self.history[-6:]:
            messages.append({'role': h['role'], 'content': h['content']})
        messages.append({'role': 'user', 'content': user_message if user_message else 'Hi tutor, let us start!'})

        reply = self.brain.generate(messages, socratic_system_prompt)
        
        if user_message:
            self.history.append({'role': 'user', 'content': user_message})
        self.history.append({'role': 'assistant', 'content': reply})
        
        token_stats = self.tracker.get_stats(self.brain.active_provider)
        token_stats['current_concept'] = current_node.title
        token_stats['mastery_pct'] = mastery_pct
        
        return reply, token_stats


app = FastAPI(title="Personal AI Tutor with Token Warning System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tutor = PersonalAITutor()

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal AI Tutor - Socratic Math & Token HUD</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --primary: #38bdfx;
            --accent: #818cf8;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --warning-bg: #451a03;
            --warning-border: #f59e0b;
        }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }
        #sidebar {
            width: 320px;
            background: var(--card-bg);
            border-right: 1px solid #334155;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        #chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: relative;
        }
        #chat-header {
            padding: 20px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            max-width: 75%;
            padding: 14px 18px;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 15px;
        }
        .message.user {
            background: #0284c7;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }
        .message.tutor {
            background: #334155;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
            border-left: 4px solid var(--primary);
        }
        #input-container {
            padding: 20px;
            background: #1e293b;
            border-top: 1px solid #334155;
            display: flex;
            gap: 12px;
        }
        input[type="text"] {
            flex: 1;
            background: #0f172a;
            border: 1px solid #475569;
            color: #fff;
            padding: 14px 18px;
            border-radius: 8px;
            font-size: 15px;
            outline: none;
        }
        button {
            background: var(--primary);
            color: #0f172a;
            font-weight: 600;
            border: none;
            padding: 0 24px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:huover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        /* FLOATING WARNING & TOKEN HUD WIDGET */
        #floating-hud {
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 340px;
            background: rgba(30, 41, 59, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid #475569;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            z-index: 9999;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        #hud-header {
            background: #0f172a;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            cursor: pointer;
        }
        .hud-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 8px #22c55e;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
            70% {
 transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
            100% {
 transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }
        #hud-body {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .token-row {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: var(--text-sub);
        }
        .token-val {
            font-weight: 700;
            color: var(--text-main);
        }
        .warning-banner {
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            color: #fef3c7;
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 12px;
            line-height: 1.4;
            display: none;
        }
        .provider-badge {
            display: inline-block;
            background: #0284c7;
            color: #fff;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 6px;
        }
        .progress-bar-bg {
            background: #334155;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #38bdfx, #818cf8);
            height: 100%;
            width: 20%;
            transition: width 0.4s ease;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <h2 style="margin:0; font-size: 20px; color: var(--primary);">Ô  Socratic Tutor</h2>
        <div style="font-size: 13px; color: var(--text-sub);">Adaptive Math Learning with Bayesian Knowledge Tracing.</div>
        
        <hr style="border-color: #334155; margin: 5px 0;">
        
        <div>
            <div style="font-size: 12px; color: var(--text-sub); text-transform: uppercase; font-weight:700;">Current Topic</div>
            <div id="sidebar-concept" style="font-size: 16px; font-weight:700; margin-top: 4px;">Number Line</div>
        </div>

        <div>
            <div style="font-size: 12px; color: var(--text-sub); text-transform: uppercase; font-weight:700; margin-bottom: 6px;">BKT Mastery Progress</div>
            <div class="progress-bar-bg">
                <div id="mastery-fill" class="progress-bar-fill"></div>
            </div>
            <div id="sidebar-mastery" style="font-size: 12px; color: var(--primary); font-weight:700; margin-top: 6px;">Mastery: 20%</div>
        </div>
    </div>

    <div id="chat-container">
        <div id="chat-header">
            <div>
                <strong id="header-topic">Algebra — Number Line and Absolute Value</strong>
                <div style="font-size: 12px; color: var(--text-sub);">Pedagogy: Guided Socratic Questions (No answer dumps)</div>
            </div>
        </div>

        <div id="messages">
            <div class="message tutor">Hello! I'm your Socratic Math Tutor. Let's start with understanding positive and negative positions on a number line.<br><br>Can you explain what a number line is in your own words?</div>
        </div>

        <div id="input-container">
            <input type="text" id="user-input" placeholder="Type your answer or question here..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <!-- BLOATING WARNING & TOKEN HUD WIDGET -->
    <div id="floating-hud">
        <div id="hud-header" onclick="toggleHud()">
            <div class="hud-title">
                <div class="pulse-dot" id="status-dot"></div>
                Token & API Monitor
            </div>
            <span class="provider-badge" id="hud-provider">Hugging Face (Qwen-72B)</span>
        </div>
        <div id="hud-body">
            <div id="hud-warning" class="warning-banner"></div>
            
            <div class="token-row">
                <span>Last Turn Tokens:</span>
                <span class="token-val" id="hud-last-turn">0</span>
            </div>
            <div class="token-row">
                <span>Session Total Tokens:</span>
                <span class="token-val" id="hud-session-total">0</span>
            </div>
            <div class="token-row">
                <span>Estimated HF Free Rate Limit:</span>
                <span class="token-val" id="hud-remaining">4,000 t/min</span>
            </div>
        </div>
    </div>

    <script>
        let isMinimized = false;

        function toggleHud() {
            const body = document.getElementById('hud-body');
            isMinimized = !isMinimized;
            body.style.display = isMinimized ? 'none' : 'flex';
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text) return;

            const msgs = document.getElementById('messages');
            msgs.innerHTML += `<div class="message user">${text}</div>`;
            input.value = '';
            msgs.scrollTop = msgs.scrollHeight;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();

                msgs.innerHTML += `<div class="message tutor">${data.reply}</div>`;
                msgs.scrollTop = msgs.scrollHeight;

                updateHud(data.stats);
            } catch (err) {
                console.error(err);
            }
        }

        function updateHud(stats) {
            document.getElementById('hud-provider').innerText = stats.active_provider;
            document.getElementById('hud-last-turn').innerText = stats.last_turn_total + ' (' + stats.last_prompt_tokens + ' in / ' + stats.last_completion_tokens + ' out)';
            document.getElementById('hud-session-total').innerText = stats.session_total_tokens.toLocaleString();
            document.getElementById('hud-remaining').innerText = stats.estimated_remaining_hf_rate.toLocaleString() + ' t/min';

            const warnBanner = document.getElementById('hud-warning');
            const dot = document.getElementById('status-dot');

            if (stats.warnings && stats.warnings.length > 0) {
                warnBanner.style.display = 'block';
                warnBanner.innerText = stats.warnings.join(' ');
                dot.style.background = '#f59e0b';
                dot.style.boxShadow = '0 0 8px #f59e0b';
            } else {
                warnBanner.style.display = 'none';
                dot.style.background = '#22c55e';
                dot.style.boxShadow = '0 0 8px #22c55e';
            }

            if (stats.mastery_pct) {
                document.getElementById('mastery-fill').style.width = stats.mastery_pct + '%';
                document.getElementById('sidebar-mastery').innerText = 'Mastery: ' + stats.mastery_pct + '%';
            }
            if (stats.current_concept) {
                document.getElementById('sidebar-concept').innerText = stats.current_concept;
                document.getElementById('header-topic').innerText = 'Algebra — ' + stats.current_concept;
            }
        }

        fetch('/api/status').then(r=>r.json()).then(stats => updateHud(stats));
    </script>
</body>
</html>
'''

@app.get('/', response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE

@app.get('/api/status')
def get_status():
    return tutor.tracker.get_stats(tutor.brain.active_provider)

@app.post('/api/chat')
async def chat_endpoint(req: Request):
    data = await req.json()
    msg = data.get('message', '')
    reply, stats = tutor.chat_turn(msg)
    return {'reply': reply, 'stats': stats}

if __name__ == '__main__':
    print('Starting Web Tutor Server on http://localhost:8000')
    uvicorn.run(app, host='0.0.0.0', port=8000)
