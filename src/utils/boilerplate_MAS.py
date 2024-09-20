import asyncio
import logging
from typing import Dict, Any, List, Optional, Literal
from anthropic import AsyncAnthropic
import json
import os
from dotenv import load_dotenv
import time
from functools import wraps
import re
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 5
RETRY_DELAY = 60  # seconds
API_RATE_LIMIT = 1  # requests per second

# Define the available models as a Literal type
AvailableModel = Literal[
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307"
]

class AvailableModelsConfig(BaseModel):
    models: List[AvailableModel] = Field(
        default=[
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307"
        ]
    )

# Create an instance of the config
AVAILABLE_MODELS_CONFIG = AvailableModelsConfig()

# Use the validated models list
AVAILABLE_MODELS = AVAILABLE_MODELS_CONFIG.models

class RateLimiterModel(BaseModel):
    rate_limit: float = Field(..., gt=0)

class RateLimiter:
    def __init__(self, rate_limit: float):
        validated_data = RateLimiterModel.model_validate({"rate_limit": rate_limit})
        self.rate_limit = validated_data.rate_limit
        self.tokens = self.rate_limit
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

class BaseAgentModel(BaseModel):
    client: AsyncAnthropic

class BaseAgent:
    def __init__(self, client: AsyncAnthropic):
        validated_data = BaseAgentModel.model_validate({"client": client})
        self.client = validated_data.client

    @rate_limited
    async def call_api(self, system_prompt: str, prompt: str, model: AvailableModel = "claude-3-5-sonnet-20240620", max_tokens: int = 1000, temperature: float = 0.0) -> str:
        class CallAPIModel(BaseModel):
            system_prompt: str
            prompt: str
            model: AvailableModel
            max_tokens: int = Field(..., gt=0)
            temperature: float = Field(..., ge=0, le=1)

        try:
            validated_data = CallAPIModel.model_validate({
                "system_prompt": system_prompt,
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature
            })
        except ValidationError as e:
            logger.error(f"Validation error in call_api: {e}")
            raise

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=validated_data.model,
                    max_tokens=validated_data.max_tokens,
                    temperature=validated_data.temperature,
                    system=validated_data.system_prompt,
                    messages=[{"role": "user", "content": validated_data.prompt}]
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
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
            logger.error(f"Failed to extract valid JSON from response: {response[:100]}...")
            return {}

class PlannerAgentModel(BaseModel):
    client: AsyncAnthropic

class PlannerAgent(BaseAgent):
    def __init__(self, client: AsyncAnthropic):
        validated_data = PlannerAgentModel.model_validate({"client": client})
        super().__init__(validated_data.client)

    async def create_plan(self, task: str) -> Dict[str, Any]:
        class CreatePlanModel(BaseModel):
            task: str

        try:
            validated_data = CreatePlanModel.model_validate({"task": task})
        except ValidationError as e:
            logger.error(f"Validation error in create_plan: {e}")
            raise

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
        prompt = f"Create a detailed plan for the following task: {validated_data.task}"
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-opus-20240229", temperature=0.3)
        
        result = self.extract_json_data(response)
        if not result:
            logger.warning("Failed to create a valid plan. Returning a default plan.")
            return {"plan": [{"step_number": 1, "description": "Execute the task", "expected_outcome": "Task completed", "assigned_model": "claude-3-5-sonnet-20240620"}]}
        
        # Validate the assigned models in the plan
        for step in result.get('plan', []):
            if step.get('assigned_model') not in AVAILABLE_MODELS:
                logger.warning(f"Invalid model {step.get('assigned_model')} assigned. Using default model.")
                step['assigned_model'] = "claude-3-5-sonnet-20240620"
        
        logger.info(f"Created plan: {result}")
        return result

class ExecutorAgentModel(BaseModel):
    client: AsyncAnthropic

class ExecutorAgent(BaseAgent):
    def __init__(self, client: AsyncAnthropic):
        validated_data = ExecutorAgentModel.model_validate({"client": client})
        super().__init__(validated_data.client)

    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        class ExecuteStepModel(BaseModel):
            step: Dict[str, Any]

        try:
            validated_data = ExecuteStepModel.model_validate({"step": step})
        except ValidationError as e:
            logger.error(f"Validation error in execute_step: {e}")
            raise

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
        prompt = f"Execute and report on the following step: {json.dumps(validated_data.step)}"
        
        model = validated_data.step.get("assigned_model", "claude-3-5-sonnet-20240620")
        if model not in AVAILABLE_MODELS:
            logger.warning(f"Invalid model {model} assigned. Using default model.")
            model = "claude-3-5-sonnet-20240620"
        
        response = await self.call_api(system_prompt, prompt, model=model, temperature=0.2)
        
        result = self.extract_json_data(response)
        if not result:
            logger.warning(f"Failed to execute step {validated_data.step.get('step_number', 'unknown')}. Returning a default result.")
            return {
                "step_number": validated_data.step.get("step_number", "unknown"),
                "action_taken": "Failed to execute step",
                "outcome": "Execution failed due to invalid response",
                "success": False,
                "challenges": ["Invalid response format"],
                "next_steps": ["Retry execution with simplified input"]
            }
        
        logger.info(f"Executed step: {result}")
        return result

class SynthesizerAgentModel(BaseModel):
    client: AsyncAnthropic

class SynthesizerAgent(BaseAgent):
    def __init__(self, client: AsyncAnthropic):
        validated_data = SynthesizerAgentModel.model_validate({"client": client})
        super().__init__(validated_data.client)

    async def synthesize(self, task: str, execution_results: List[Dict[str, Any]], output_file: str) -> str:
        class SynthesizeModel(BaseModel):
            task: str
            execution_results: List[Dict[str, Any]]
            output_file: str

        try:
            validated_data = SynthesizeModel.model_validate({
                "task": task,
                "execution_results": execution_results,
                "output_file": output_file
            })
        except ValidationError as e:
            logger.error(f"Validation error in synthesize: {e}")
            raise

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
        
        Original Task: {validated_data.task}
        
        Execution Results:
        {json.dumps(validated_data.execution_results, indent=2)}
        
        Provide a detailed analysis of these results, including an assessment of overall success, key findings, challenges encountered, and recommendations for future actions.
        """
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-opus-20240229", max_tokens=4000, temperature=0.2)
        
        try:
            with open(validated_data.output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Synthesis report written to {validated_data.output_file}")
            return validated_data.output_file
        except IOError as e:
            error_message = f"Failed to write synthesis report: {str(e)}"
            logger.error(error_message)
            return "synthesis_error.txt"

class OrchestratorModel(BaseModel):
    api_key: str

class Orchestrator:
    def __init__(self, api_key: str):
        validated_data = OrchestratorModel.model_validate({"api_key": api_key})
        self.client = AsyncAnthropic(api_key=validated_data.api_key)
        self.planner = PlannerAgent(self.client)
        self.executor = ExecutorAgent(self.client)
        self.synthesizer = SynthesizerAgent(self.client)

    async def execute_task(self, task: str, output_file: str = "synthesis_report.txt") -> Dict[str, Any]:
        class ExecuteTaskModel(BaseModel):
            task: str
            output_file: str

        try:
            validated_data = ExecuteTaskModel.model_validate({
                "task": task,
                "output_file": output_file
            })
        except ValidationError as e:
            logger.error(f"Validation error in execute_task: {e}")
            raise

        logger.info(f"Starting execution of task: {validated_data.task}")
        
        try:
            plan = await self.planner.create_plan(validated_data.task)
            
            execution_results = []
            for step in plan.get('plan', []):
                step_result = await self.executor.execute_step(step)
                execution_results.append(step_result)
            
            synthesis_file = await self.synthesizer.synthesize(validated_data.task, execution_results, validated_data.output_file)
            
            final_result = {
                "task": validated_data.task,
                "plan": plan,
                "execution_results": execution_results,
                "synthesis_report_file": synthesis_file,
                "status": "completed"
            }
            
            logger.info(f"Task execution and synthesis completed: {validated_data.task}")
            return final_result
        
        except Exception as e:
            logger.error(f"An error occurred during task execution: {str(e)}", exc_info=True)
            return {
                "task": validated_data.task,
                "status": "error",
                "error_message": str(e)
            }

async def main():
    load_dotenv(override=True)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.error("API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return
    
    class MainConfigModel(BaseModel):
        api_key: str
        task: str
        output_file: str

    try:
        config = MainConfigModel.model_validate({
            "api_key": api_key,
            "task": "Analyze the impact of artificial intelligence on job markets in the next decade.",
            "output_file": "ai_impact_analysis.txt"
        })
    except ValidationError as e:
        logger.error(f"Validation error in main configuration: {e}")
        return

    orchestrator = Orchestrator(config.api_key)
    
    try:
        result = await orchestrator.execute_task(config.task, config.output_file)
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