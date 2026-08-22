import os
import sys
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

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

class LLMBrain:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')
        self.hf_models = [
            'Qwen/Qwen2.5-72B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct',
            'Qwen/Qwen2.5-Coder-32B-Instruct'
        ]
        self.active_provider = 'Initializing...'

    def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        full_messages = [{'role': 'system', 'content': system_prompt}] + messages
        
        # 1. Try OpenAI if key is present
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=full_messages,
                    temperature=0.7,
                    max_tokens=400
                )
                self.active_provider = 'OpenAI (gpt-4o-mini)'
                return response.choices[0].message.content.strip()
            except Exception as e:
                pass

        # 2. Fallback to Hugging Face models
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
                    self.active_provider = 'Hugging Face (' + m + ')'
                    return response.choices[0].message.content.strip()
                except Exception as hf_err:
                    continue

        return 'Model unavailable. Please check API keys.'

class PersonalAITutor:
    def __init__(self, curriculum: List[ConceptNode]):
        self.curriculum = {c.id: c for c in curriculum}
        self.mastery: Dict[str, float] = {c.id: 0.20 for c in curriculum}
        self.history: List[Dict] = []
        self.brain = LLMBrain()
        self.current_concept_id: str = curriculum[0].id

    def get_next_concept(self) -> Optional[ConceptNode]:
        for c in self.curriculum.values():
            if self.mastery[c.id] < 0.85:
                if all(self.mastery[p] >= 0.70 for p in c.prereqs):
                    return c
        return None

    def evaluate_response(self, learner_input: str, concept: ConceptNode) -> bool:
        eval_prompt = 'Concept: ' + concept.title + '. Student answer: ' + repr(learner_input) + '. Is this answer mathematically correct or showing correct reasoning? Reply with ONLY CORRECT or INCORRECT.'
        res = self.brain.generate([{'role': 'user', 'content': eval_prompt}], 'You evaluate math answers.')
        return 'CORRECT' in res.upper()

    def chat_turn(self, user_message: str) -> str:
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
        
        return reply

def run_cli_session():
    tutor = PersonalAITutor(ALGEBRA_CURRICULUM)
    print('=' * 72)
    print(' PERSONAL AI TUTOR (Socratic Pedagogy + BKT Learner Model)')
    print('=' * 72)
    
    greeting = tutor.chat_turn('')
    print(f'\n[Active Brain: {tutor.brain.active_provider}]')
    print('AI Tutor:\n' + greeting + '\n')

if __name__ == '__main__':
    run_cli_session()
