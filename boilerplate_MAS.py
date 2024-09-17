import asyncio
import logging
from typing import Dict, Any, List
from anthropic import AsyncAnthropic
import json
import os
from dotenv import load_dotenv
import time
from functools import wraps
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class RateLimiter:
    """
    A rate limiter to control the frequency of API calls.
    
    This class implements a token bucket algorithm to limit the rate of API calls.
    It ensures that calls are made at most at the specified rate limit.
    """

    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.updated_at = time.time()

    async def acquire(self):
        """
        Acquire a token to make an API call.
        
        This method will delay the execution if the rate limit has been exceeded,
        ensuring that subsequent calls adhere to the specified rate limit.
        """
        now = time.time()
        time_passed = now - self.updated_at
        self.tokens = min(self.rate_limit, self.tokens + time_passed * self.rate_limit)
        self.updated_at = now
        if self.tokens < 1:
            await asyncio.sleep((1 - self.tokens) / self.rate_limit)
        self.tokens -= 1

def rate_limited(func):
    """
    A decorator to apply rate limiting to a function.
    
    This decorator ensures that the decorated function is called at most
    at the rate specified by API_RATE_LIMIT.
    """
    limiter = RateLimiter(API_RATE_LIMIT)
    @wraps(func)
    async def wrapper(*args, **kwargs):
        await limiter.acquire()
        return await func(*args, **kwargs)
    return wrapper

class BaseAgent:
    def __init__(self, client: AsyncAnthropic):
        self.client = client

    @rate_limited
    async def call_api(self, system_prompt: str, prompt: str, model: str = "claude-3-sonnet-20240320", max_tokens: int = 1000, temperature: float = 0.0) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"API call failed, retrying in {RETRY_DELAY} seconds. Error: {str(e)}")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(f"API call failed after {MAX_RETRIES} attempts. Error: {str(e)}")
                    raise

    def extract_json_data(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON data from the API response, with improved error handling.
        """
        try:
            # Try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to find a JSON object within the response
            import re
            json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, log the error and return an empty dict
            logger.error(f"Failed to extract valid JSON from response: {response[:100]}...")
            return {}

class PlannerAgent(BaseAgent):
    async def create_plan(self, task: str) -> Dict[str, Any]:
        system_prompt = f"""
        You are a planning expert. Create a structured plan for the given task.
        Your response must be a valid JSON object with the following structure:
        {{
            "plan": [
                {{
                    "step_number": 1,
                    "description": "Step description",
                    "expected_outcome": "Expected outcome of this step",
                    "assigned_model": "Model to use for this step"
                }},
                ...
            ]
        }}
        Ensure your response can be parsed as JSON. Do not include any text outside the JSON structure.
        For the "assigned_model" field, choose from the following models based on the step's complexity and requirements:
        {AVAILABLE_MODELS}
        Assign models strategically, considering that Opus is most capable but slowest, Haiku is fastest but least capable, and Sonnet is a balance between the two.
        """
        prompt = f"Create a detailed plan for the following task: {task}"
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-opus-20240229", temperature=0.3)
        
        result = self.extract_json_data(response)
        if not result:
            logger.warning("Failed to create a valid plan. Returning a default plan.")
            return {"plan": [{"step_number": 1, "description": "Execute the task", "expected_outcome": "Task completed", "assigned_model": "claude-3-sonnet-20240320"}]}
        
        logger.info(f"Created plan: {result}")
        return result

class ExecutorAgent(BaseAgent):
    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
        You are an expert at executing tasks. Perform the given step and report the results.
        Your response must be a valid JSON object with the following structure:
        {
            "step_number": <step number>,
            "action_taken": "Description of the action taken",
            "outcome": "Detailed outcome of the action",
            "success": true/false,
            "challenges": ["List of any challenges encountered"],
            "next_steps": ["Suggested next steps or considerations"]
        }
        Ensure your response can be parsed as JSON. Do not include any text outside the JSON structure.
        """
        prompt = f"Execute and report on the following step: {json.dumps(step)}"
        
        model = step.get("assigned_model", "claude-3-sonnet-20240320")
        response = await self.call_api(system_prompt, prompt, model=model, temperature=0.2)
        
        result = self.extract_json_data(response)
        if not result:
            logger.warning(f"Failed to execute step {step.get('step_number', 'unknown')}. Returning a default result.")
            return {
                "step_number": step.get("step_number", "unknown"),
                "action_taken": "Failed to execute step",
                "outcome": "Execution failed due to invalid response",
                "success": False,
                "challenges": ["Invalid response format"],
                "next_steps": ["Retry execution with simplified input"]
            }
        
        logger.info(f"Executed step: {result}")
        return result

class SynthesizerAgent(BaseAgent):
    async def synthesize(self, task: str, execution_results: List[Dict[str, Any]], output_file: str) -> str:
        system_prompt = """
        You are an expert at synthesizing information and writing comprehensive reports.
        Analyze the results from multiple executed steps and provide a detailed, well-structured report.
        Your report should be clear, concise, and suitable for a professional audience.
        
        Structure your report with the following sections:
        1. Executive Summary
        2. Introduction
        3. Methodology
        4. Results and Analysis
        5. Key Findings
        6. Challenges and Limitations
        7. Recommendations
        8. Conclusion
        
        Use appropriate formatting, such as headers, bullet points, and paragraphs to enhance readability.
        Ensure that your report is cohesive, flowing logically from one section to the next.
        """
        prompt = f"""
        Synthesize the results of the following task execution into a comprehensive report:
        
        Original Task: {task}
        
        Execution Results:
        {json.dumps(execution_results, indent=2)}
        
        Provide a detailed analysis of these results, including an assessment of overall success, key findings, challenges encountered, and recommendations for future actions.
        """
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-opus-20240229", max_tokens=4000, temperature=0.2)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Synthesis report written to {output_file}")
            return output_file
        except IOError as e:
            error_message = f"Failed to write synthesis report: {str(e)}"
            logger.error(error_message)
            return "synthesis_error.txt"

class Orchestrator:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.planner = PlannerAgent(self.client)
        self.executor = ExecutorAgent(self.client)
        self.synthesizer = SynthesizerAgent(self.client)

    async def execute_task(self, task: str, output_file: str = "synthesis_report.txt") -> Dict[str, Any]:
        logger.info(f"Starting execution of task: {task}")
        
        try:
            plan = await self.planner.create_plan(task)
            
            execution_results = []
            for step in plan.get('plan', []):
                step_result = await self.executor.execute_step(step)
                execution_results.append(step_result)
            
            synthesis_file = await self.synthesizer.synthesize(task, execution_results, output_file)
            
            final_result = {
                "task": task,
                "plan": plan,
                "execution_results": execution_results,
                "synthesis_report_file": synthesis_file,
                "status": "completed"
            }
            
            logger.info(f"Task execution and synthesis completed: {task}")
            return final_result
        
        except Exception as e:
            logger.error(f"An error occurred during task execution: {str(e)}", exc_info=True)
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
    
    orchestrator = Orchestrator(api_key)
    task = "Analyze the impact of artificial intelligence on job markets in the next decade."
    output_file = "ai_impact_analysis.txt"
    
    try:
        result = await orchestrator.execute_task(task, output_file)
        print(json.dumps(result, indent=2))
        
        if result.get('status') == 'completed':
            print(f"\nSynthesis report has been written to: {result.get('synthesis_report_file', 'N/A')}")
            
            synthesis_file = result.get('synthesis_report_file')
            if synthesis_file and os.path.exists(synthesis_file):
                with open(synthesis_file, 'r') as f:
                    print("\nSynthesis Report Contents:")
                    print(f.read())
            else:
                print("\nSynthesis report file not found or not generated.")
        else:
            print("\nTask execution failed. No synthesis report generated.")
    except Exception as e:
        logger.error(f"An unhandled error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())