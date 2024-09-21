import asyncio
import logging
from typing import Dict, Any, Protocol, runtime_checkable, List, Optional,Literal,Union
from anthropic import AsyncAnthropic
import json
import os
from dotenv import load_dotenv
import time
from functools import wraps
import re
from pydantic import BaseModel, Field, ConfigDict,ValidationError

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

# Protocols
@runtime_checkable
class UserInputProtocol(Protocol):
    async def get_user_input(self, prompt: str) -> str:
        ...

@runtime_checkable
class SimilarityCheckerProtocol(Protocol):
    async def check_similarity(self, new_question: str, cached_questions: List[str]) -> Optional[str]:
        ...

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

# configuration models

class BaseAgentConfig(BaseModel):
    max_retries: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1.0, ge=0)

class APIConfig(BaseModel):
    model: str = Field(default="claude-3-5-sonnet-20240620")
    temperature: float = Field(default=0.0, ge=0, le=1)

    @property
    def max_tokens(self) -> int:
        MODEL_MAX_TOKENS = {
            "claude-3-opus-20240229": 4000,
            "claude-3-5-sonnet-20240620": 8000,
            "claude-3-haiku-20240307": 4000
        }
        return MODEL_MAX_TOKENS.get(self.model, 4000)

class RateLimiterModel(BaseModel):
    rate_limit: float = Field(..., gt=0)

class RateLimiter:
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.semaphore = asyncio.Semaphore(calls)
        self.task_queue = asyncio.Queue()

    async def acquire(self):
        await self.semaphore.acquire()
        asyncio.create_task(self.release_after_delay())

    async def release_after_delay(self):
        await asyncio.sleep(self.period)
        self.semaphore.release()

def rate_limited(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        await self.rate_limiter.acquire()
        return await func(self, *args, **kwargs)
    return wrapper

class ClarificationQuestion(BaseModel):
    question: str
    reason: str

class ClarificationQuestions(BaseModel):
    questions: List[ClarificationQuestion]

class ClarificationResponse(BaseModel):
    question: str
    answer: str

class ClarificationResponses(BaseModel):
    responses: List[ClarificationResponse]

class StructuredPrompt(BaseModel):
    system_prompt: str = Field(..., description="The system prompt providing context and guidelines")
    prompt: str = Field(..., description="The specific task prompt")
    output_model: str = Field(..., description="Pydantic model structure for the expected output")
    model: str = Field(default="claude-3-5-sonnet-20240620", description="The AI model to use")
    temperature: float = Field(default=0.4, ge=0, le=1, description="Temperature setting for the AI model")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens for the response")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Sequences that will stop the AI's response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the prompt")

    def format_prompt(self, **kwargs) -> 'StructuredPrompt':
        new_prompt = self.model_copy(deep=True)
        new_prompt.prompt = new_prompt.prompt.format(**kwargs)
        return new_prompt

    def to_api_parameters(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": self.prompt}]
        }

class PromptCatalog:
    @staticmethod
    def create_prompts_prompt() -> StructuredPrompt:
        return StructuredPrompt(
            system_prompt="""
            You are an expert prompt engineer working for Anthropic. Create a system prompt and a task prompt for a helper agent based on the given domain, task, and required expertise.
            Your response must be a valid JSON object that conforms to the Pydantic model structure provided in the output_model field.
            The system prompt should provide context and guidelines for the helper agent's role and capabilities.
            The task prompt should clearly articulate the specific task to be performed and instruct the agent to respond in a structured JSON format.
            """,
            prompt="""
            Create prompts for a helper agent with the following details:
            Domain: {domain}
            Task: {task}
            Required Expertise: {required_expertise}
            Ensure that the task prompt explicitly instructs the helper agent to provide its response as a structured JSON object with clearly defined keys relevant to the task.
            """,
            output_model="""
            class PromptOutput(BaseModel):
                system_prompt: str
                task_prompt: str
            """,
            metadata={
                "purpose": "Generate system and task prompts for helper agents",
                "version": "1.0"
            }
        )

# Agent Models

class BaseAgentModel(BaseModel):
    client: AsyncAnthropic
    config: BaseAgentConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

class UserInteractionManagerModel(BaseModel):
    similarity_checker: SimilarityCheckerProtocol
    base_agent: 'BaseAgent'
    question_cache: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class SimilarityCheckerModel(BaseModel):
    base_agent: 'BaseAgent'

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

class PromptAgent(BaseAgent):
    async def create_prompts(self, domain: str, task: str, required_expertise: str) -> Dict[str, str]:
        prompt_template = PromptCatalog.create_prompts_prompt()
        formatted_prompt = prompt_template.format_prompt(
            domain=domain,
            task=task,
            required_expertise=required_expertise
        )
        
        logger.debug(f"PromptAgent - Input: Domain: {domain}, Task: {task}, Expertise: {required_expertise}")
        
        api_params = formatted_prompt.to_api_parameters()
        response = await self.call_api(**api_params)
        
        logger.debug(f"PromptAgent - Raw Response: {response[:500]}...")
        
        try:
            prompts_dict = json.loads(response)
            # Here we could use Pydantic to validate the output if needed
            logger.debug(f"PromptAgent - Extracted Prompts: {json.dumps(prompts_dict, indent=2)}")
            return prompts_dict
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing prompts: {e}. Using default prompts.")
            return {
                "system_prompt": f"You are an expert in {domain} with expertise in {required_expertise}. Provide detailed and structured responses.",
                "task_prompt": f"Complete the following task: {task}. Provide your response in a structured JSON format."
            }
        
class QuestionCacheEntry(BaseModel):
    question: str
    answer: str

class GlobalQuestionCache(BaseModel):
    cache: Dict[str, QuestionCacheEntry] = Field(default_factory=dict)

    def add(self, question: str, answer: str):
        self.cache[question] = QuestionCacheEntry(question=question, answer=answer)

    def get(self, question: str) -> Optional[str]:
        entry = self.cache.get(question)
        return entry.answer if entry else None

    def get_all(self) -> Dict[str, str]:
        return {k: v.answer for k, v in self.cache.items()}

    def get_all_questions(self) -> List[str]:
        return list(self.cache.keys())

class SimilarityCheckerConfig(BaseModel):
    model: str = Field(default="claude-3-5-sonnet-20240620")
    max_tokens: int = Field(default=2000,ge=500,le=4000)
    temperature: float = Field(default=0.0, ge=0, le=1)
    rate_limit_calls: int = Field(default=5,ge=0,le=10)
    rate_limit_period: float = Field(default=1.0,ge=1.0,le=5.0)

class SimilarityChecker(BaseModel):
    client: AsyncAnthropic
    config: SimilarityCheckerConfig = Field(default_factory=SimilarityCheckerConfig)
    rate_limiter: RateLimiter = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_calls, 
            self.config.rate_limit_period
        )

    @rate_limited
    async def check_similarity(self, new_question: str, cached_questions: List[str]) -> Optional[str]:
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
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip()
            return result if "No similar question found" not in result else None
        except Exception as e:
            logger.error(f"Error in check_similarity: {e}")
            return None
        
class UserInteractionManager(BaseModel):
    global_cache: GlobalQuestionCache
    similarity_checker: SimilarityChecker

    class Config:
        arbitrary_types_allowed = True

    async def get_user_input(self, prompt: str) -> str:
        similar_question = await self.similarity_checker.check_similarity(
            prompt, 
            self.global_cache.get_all_questions()
        )
        
        if similar_question:
            cached_response = self.global_cache.get(similar_question)
            print(f"Found similar question: '{similar_question}'")
            print(f"Using cached response: '{cached_response}'")
            return cached_response

        # In a real async environment, you might want to use an async input method
        response = input(prompt + "\nYour response: ")
        self.global_cache.add(prompt, response)
        return response
    

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
        Assign models strategically, considering that Opus is most capable but slowest, Haiku is fastest but least capable, and Sonnet is a balance between the two that also incorporates caching.
        """
        prompt = f"Create a detailed plan for the following task: {validated_data.task}"
        
        response = await self.call_api(system_prompt, prompt, model="claude-3-5-sonnet-20240620", temperature=0.3)
        
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