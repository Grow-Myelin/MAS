import asyncio
import logging
from typing import Dict, Any, List, Union
from anthropic import AsyncAnthropic
import json
import os
from dotenv import load_dotenv
import time
from functools import wraps
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
AVAILABLE_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307"
]
MAX_RETRIES = 5
RETRY_DELAY = 60  # seconds
API_RATE_LIMIT = 1  # requests per second

MODEL_MAX_TOKENS = {
    "claude-3-opus-20240229": 4000,
    "claude-3-5-sonnet-20240620": 8000,
    "claude-3-haiku-20240307": 4000
}

class RateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.updated_at = time.time()

    async def acquire(self):
        now = time.time()
        time_passed = now - self.updated_at
        self.tokens = min(self.rate_limit, self.tokens + time_passed * self.rate_limit)
        self.updated_at = now
        if self.tokens < 1:
            await asyncio.sleep((1 - self.tokens) / self.rate_limit)
        self.tokens -= 1

def rate_limited(func):
    limiter = RateLimiter(API_RATE_LIMIT)
    @wraps(func)
    async def wrapper(*args, **kwargs):
        await limiter.acquire()
        return await func(*args, **kwargs)
    return wrapper

class GlobalQuestionCache:
    def __init__(self):
        self.cache = {}

    def add(self, question: str, answer: str):
        self.cache[question] = answer

    def get(self, question: str) -> Union[str, None]:
        return self.cache.get(question)

    def get_all(self) -> Dict[str, str]:
        return self.cache

    def get_all_questions(self) -> List[str]:
        return list(self.cache.keys())

class SimilarityChecker:
    def __init__(self, client: AsyncAnthropic):
        self.client = client

    @rate_limited
    async def check_similarity(self, new_question: str, cached_questions: List[str]) -> Union[str, None]:
        system_prompt = """
        You are a quick-thinking AI assistant specializing in identifying similar questions.
        Your task is to determine if a new question is similar to any of the previously asked questions.
        If you find a similar question, return its exact text. If not, return "No similar question found."
        Respond only with the similar question or the "No similar question found" message.
        """
        prompt = f"""
        New question: {new_question}

        Previously asked questions:
        {json.dumps(cached_questions, indent=2)}

        Is the new question similar to any of the previously asked questions?
        """

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.content[0].text.strip()
        return result if "No similar question found" not in result else None

class UserInteractionManager:
    def __init__(self, global_cache: GlobalQuestionCache, similarity_checker: SimilarityChecker):
        self.global_cache = global_cache
        self.similarity_checker = similarity_checker

    async def get_user_input(self, prompt: str) -> str:
        similar_question = await self.similarity_checker.check_similarity(prompt, self.global_cache.get_all_questions())
        
        if similar_question:
            cached_response = self.global_cache.get(similar_question)
            print(f"Found similar question: '{similar_question}'")
            print(f"Using cached response: '{cached_response}'")
            return cached_response

        response = input(prompt + "\nYour response: ")
        self.global_cache.add(prompt, response)
        return response

class BaseAgent:
    def __init__(self, client: AsyncAnthropic, user_interaction_manager: UserInteractionManager):
        self.client = client
        self.user_interaction_manager = user_interaction_manager

    @rate_limited
    async def call_api(self, system_prompt: str, prompt: str, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = None, temperature: float = 0.0) -> str:
        if max_tokens is None:
            max_tokens = MODEL_MAX_TOKENS.get(model, 4096)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                logger.debug(f"API Response: {response.content[0].text[:500]}...")
                return response.content[0].text
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"API call failed, retrying in {RETRY_DELAY} seconds. Error: {str(e)}")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(f"API call failed after {MAX_RETRIES} attempts. Error: {str(e)}")
                    raise

    async def ask_for_clarification(self, topic: str, context: str) -> Dict[str, Any]:
        system_prompt = """
        You are an AI assistant helping to create a business plan. Your task is to identify areas in the given context that need clarification or more information from the user.
        Generate specific questions that will help gather the necessary details. Your response should be a JSON object with the following structure:
        {
            "questions": [
                {
                    "question": "The specific question to ask",
                    "reason": "Why this question is important or what information it aims to gather"
                },
                ...
            ]
        }
        Limit your response to the most critical questions. Ensure your response can be parsed as JSON.
        Before generating new questions, check the provided context for any existing answers.
        """
        prompt = f"Topic: {topic}\n\nContext: {context}\n\nBased on this information and considering any previously answered questions, what specific questions should we ask the user to gather more details or clarify any points?"

        response = await self.call_api(system_prompt, prompt, temperature=0.4)
        questions_data = self.extract_json_data(response)
        
        user_responses = {}
        for item in questions_data.get('questions', []):
            question = item['question']
            print(f"\nReason for asking: {item['reason']}")
            user_response = await self.user_interaction_manager.get_user_input(question)
            user_responses[question] = user_response

        return user_responses

    def extract_json_data(self, response: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                return data
            else:
                logger.error(f"Unexpected JSON structure. Expected dict or list, but got {type(data)}.")
                return []
        except json.JSONDecodeError:
            json_pattern = r'\[(?:[^[\]]|\[(?:[^[\]]|\[[^[\]]*\])*\])*\]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, list):
                            return [item for item in data if isinstance(item, dict)]
                        elif isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        continue
            
            logger.error(f"Failed to extract valid JSON from response. Full response: {response}")
            return []

class PromptAgent(BaseAgent):
    async def create_prompts(self, domain: str, task: str, required_expertise: str) -> Dict[str, str]:
        system_prompt = """
        You are an expert prompt engineer working for anthropic. Create a system prompt and a task prompt for a helper agent based on the given domain, task, and required expertise.
        Your response must be a valid JSON object with the following structure:
        {
            "system_prompt": "Detailed system prompt for the helper agent",
            "task_prompt": "Specific task prompt for the helper agent"
        }
        Ensure your response can be parsed as JSON. Do not include any text outside the JSON structure.
        The system prompt should provide context and guidelines for the helper agent's role and capabilities.
        The task prompt should clearly articulate the specific task to be performed and instruct the agent to respond in a structured JSON format.
        """
        prompt = f"""
        Create prompts for a helper agent with the following details:
        Domain: {domain}
        Task: {task}
        Required Expertise: {required_expertise}

        Ensure that the task prompt explicitly instructs the helper agent to provide its response as a structured JSON object with clearly defined keys relevant to the task.
        """
        
        logger.debug(f"PromptAgent - Input: Domain: {domain}, Task: {task}, Expertise: {required_expertise}")
        response = await self.call_api(system_prompt, prompt, model="claude-3-5-sonnet-20240620", temperature=0.4)
        logger.debug(f"PromptAgent - Raw Response: {response[:500]}...")
        
        prompts = self.extract_json_data(response)
        logger.debug(f"PromptAgent - Extracted Prompts: {json.dumps(prompts, indent=2)}")
        
        if not isinstance(prompts, dict):
            logger.error(f"Expected prompts to be a dictionary, but got {type(prompts)}. Using default prompts.")
            prompts = {
                "system_prompt": f"You are an expert in {domain} with expertise in {required_expertise}. Provide detailed and structured responses.",
                "task_prompt": f"Complete the following task: {task}. Provide your response in a structured JSON format."
            }
        
        return prompts

class InteractivePlanningAgent(BaseAgent):
    def __init__(self, client: AsyncAnthropic, domain: str, prompt_agent: PromptAgent, user_interaction_manager: UserInteractionManager):
        super().__init__(client, user_interaction_manager)
        self.domain = domain
        self.prompt_agent = prompt_agent

    async def create_plan(self, task: str) -> List[Dict[str, Any]]:
        system_prompt = f"""
        You are an expert planner in the domain of {self.domain}. Create a detailed plan with tasks for helper agents to execute.
        Your response must be a valid JSON array of task objects with the following structure:
        [
            {{
                "task_description": "Detailed description of the task",
                "required_expertise": "Specific expertise needed for this task",
                "assigned_model": "Model to use for this task (choose from {AVAILABLE_MODELS})"
            }},
            ...
        ]
        Assign models strategically, considering that Claude-3-Opus is the most capable but slowest, Claude-3-Haiku is the fastest but least capable, and Claude-3-5-Sonnet is a balance between the two with the ability to cache information from previous calls.
        Ensure your response can be parsed as JSON. Do not include any text outside the JSON structure.
        """
        prompt = f"Create a detailed plan for the following task in the {self.domain} domain: {task}"
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-5-sonnet-20240620", temperature=0.4)
        plan = self.extract_json_data(response)

        # Ensure plan is a list
        if not isinstance(plan, list):
            logger.error(f"Expected plan to be a list, but got {type(plan)}. Response: {response}")
            plan = []

        # Ask for clarification on each task if needed
        for i, task in enumerate(plan):
            if isinstance(task, dict):
                clarification = await self.ask_for_clarification(f"{self.domain} - {task.get('task_description', 'Unknown Task')}", json.dumps(task))
                if clarification:
                    task['user_input'] = clarification
                    # Update the task based on user input
                    update_prompt = f"Update the following task based on user input:\nOriginal task: {json.dumps(task)}\nUser input: {json.dumps(clarification)}"
                    updated_task_response = await self.call_api(system_prompt, update_prompt, model="claude-3-5-sonnet-20240620", temperature=0.3)
                    updated_task = self.extract_json_data(updated_task_response)
                    
                    # Ensure updated_task is a dictionary before updating
                    if isinstance(updated_task, dict):
                        task.update(updated_task)
                    else:
                        logger.error(f"Expected updated_task to be a dictionary, but got {type(updated_task)}. Response: {updated_task_response}")

                prompts = await self.prompt_agent.create_prompts(self.domain, task.get('task_description', ''), task.get('required_expertise', ''))
                task['system_prompt'] = prompts.get('system_prompt', f"You are an expert in {self.domain}. Provide detailed and structured responses.")
                task['task_prompt'] = prompts.get('task_prompt', f"Complete the following task: {task.get('task_description', '')}. Provide your response in a structured JSON format.")
            else:
                logger.error(f"Expected task to be a dictionary, but got {type(task)}. Task: {task}")
                plan[i] = {"task_description": str(task), "required_expertise": "Unknown", "assigned_model": "claude-3-5-sonnet-20240620"}

        return plan

class InteractiveHelperAgent(BaseAgent):
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = task.get('system_prompt', '')
        prompt = task.get('task_prompt', '')
        
        response = await self.call_api(system_prompt, prompt, model=task['assigned_model'], temperature=0.3)
        result = self.extract_json_data(response)

        # Ask for clarification if needed
        clarification = await self.ask_for_clarification(f"Task: {task['task_description']}", json.dumps(result))
        if clarification:
            update_prompt = f"Update your response based on this user input: {json.dumps(clarification)}\nOriginal response: {json.dumps(result)}"
            updated_response = await self.call_api(system_prompt, update_prompt, model=task['assigned_model'], temperature=0.3)
            result = self.extract_json_data(updated_response)

        return result

class DocumentWriter(BaseAgent):
    def __init__(self, client: AsyncAnthropic, file_name: str, user_interaction_manager: UserInteractionManager):
        super().__init__(client, user_interaction_manager)
        self.file_name = file_name

    async def write_section(self, section_title: str, content: Dict[str, Any]) -> None:
        system_prompt = """
        You are an expert technical writer. Convert the given JSON content into a well-structured Markdown section.
        Create clear headings, subheadings, and use appropriate Markdown formatting to enhance readability.
        Ensure that all information from the JSON is included and properly organized in the section.
        """
        prompt = f"Convert this JSON content into a well-structured Markdown section titled '{section_title}': {json.dumps(content)}"
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-5-sonnet-20240620", max_tokens=7500, temperature=0.2)
        
        try:
            with open(self.file_name, 'a', encoding='utf-8') as f:
                f.write(f"\n\n## {section_title}\n\n")
                f.write(response)
            logger.info(f"Section '{section_title}' written to {self.file_name}")
        except IOError as e:
            error_message = f"Failed to write section '{section_title}': {str(e)}"
            logger.error(error_message)

class InteractiveBusinessPlanningOrchestrator:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.global_question_cache = GlobalQuestionCache()
        self.similarity_checker = SimilarityChecker(self.client)
        self.user_interaction_manager = UserInteractionManager(self.global_question_cache, self.similarity_checker)
        self.document_writer = DocumentWriter(self.client, "interactive_business_plan.md", self.user_interaction_manager)
        self.prompt_agent = PromptAgent(self.client, self.user_interaction_manager)
        
        self.business_strategy_planner = InteractivePlanningAgent(self.client, "business strategy", self.prompt_agent, self.user_interaction_manager)
        self.technology_planner = InteractivePlanningAgent(self.client, "technology", self.prompt_agent, self.user_interaction_manager)
        self.financial_planner = InteractivePlanningAgent(self.client, "finance", self.prompt_agent, self.user_interaction_manager)
        
        self.helper_agent = InteractiveHelperAgent(self.client, self.user_interaction_manager)

    async def execute_domain_plan(self, planner: InteractivePlanningAgent, task: str, domain: str) -> Dict[str, Any]:
        plan = await planner.create_plan(task)
        domain_results = []
        
        for task in plan:
            result = await self.helper_agent.execute_task(task)
            domain_results.append(result)
            await self.document_writer.write_section(f"{domain.capitalize()} - {task['task_description']}", result)

        return {
            "domain": domain,
            "results": domain_results
        }

    async def create_business_plan(self, task: str) -> Dict[str, Any]:
        logger.info(f"Starting interactive business plan creation for task: {task}")
        
        try:
            # Execute plans for each domain
            strategy_results = await self.execute_domain_plan(self.business_strategy_planner, task, "Business Strategy")
            tech_results = await self.execute_domain_plan(self.technology_planner, task, "Technology")
            finance_results = await self.execute_domain_plan(self.financial_planner, task, "Finance")
            
            # Additional domains (simplified for brevity)
            marketing_prompts = await self.prompt_agent.create_prompts("Marketing", "Create a marketing plan", "marketing strategy")
            marketing_task = {
                "task_description": "Create a marketing plan",
                "required_expertise": "marketing strategy",
                "assigned_model": "claude-3-5-sonnet-20240620",
                "system_prompt": marketing_prompts.get('system_prompt', ''),
                "task_prompt": marketing_prompts.get('task_prompt', '')
            }
            marketing_results = await self.helper_agent.execute_task(marketing_task)
            await self.document_writer.write_section("Marketing Plan", marketing_results)
            
            operations_prompts = await self.prompt_agent.create_prompts("Operations", "Design operations plan", "operations management")
            operations_task = {
                "task_description": "Design operations plan",
                "required_expertise": "operations management",
                "assigned_model": "claude-3-5-sonnet-20240620",
                "system_prompt": operations_prompts.get('system_prompt', ''),
                "task_prompt": operations_prompts.get('task_prompt', '')
            }
            operations_results = await self.helper_agent.execute_task(operations_task)
            await self.document_writer.write_section("Operations Plan", operations_results)
            
            # Compile full business plan
            full_business_plan = {
                "task": task,
                "business_strategy": strategy_results,
                "technology": tech_results,
                "finance": finance_results,
                "marketing": marketing_results,
                "operations": operations_results
            }
            
            # Ask for overall clarification if needed
            clarification = await self.helper_agent.ask_for_clarification("Overall Business Plan", json.dumps(full_business_plan))
            if clarification:
                await self.document_writer.write_section("Final Clarifications and Adjustments", clarification)
                full_business_plan["final_clarifications"] = clarification

            logger.info("Interactive business plan creation completed")
            return full_business_plan
        
        except Exception as e:
            logger.error(f"An error occurred during interactive business plan creation: {str(e)}", exc_info=True)
            return {
                "task": task,
                "status": "error",
                "error_message": str(e)
            }

async def main():
    load_dotenv(override=True)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.error("API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return
    
    orchestrator = InteractiveBusinessPlanningOrchestrator(api_key)
    
    # Get the initial task description from the user
    task = await orchestrator.user_interaction_manager.get_user_input("Please describe the business plan you want to create")
    
    try:
        result = await orchestrator.create_business_plan(task)
        print(json.dumps(result, indent=2))
        
        if "error_message" not in result:
            print("\nInteractive business plan has been created in 'interactive_business_plan.md'")
            print("You can review the plan and provide feedback at each step.")
        else:
            print("\nBusiness plan creation failed. Please check the error message and logs.")
    except Exception as e:
        logger.error(f"An unhandled error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())