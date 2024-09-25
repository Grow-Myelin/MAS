# v2 = interactive_business_planner + pydantic and complications

import asyncio
import json
import logging
import os
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable,Type,TypeVar,get_type_hints
from abc import ABC, abstractmethod
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError,create_model
import traceback
import time
import re
import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
    """Protocol for getting user input."""
    
    async def get_user_input(self, prompt: str) -> str:
        """
        Get input from the user.

        Args:
            prompt (str): The prompt to display to the user.

        Returns:
            str: The user's input.
        """
        ...

@runtime_checkable
class SimilarityCheckerProtocol(Protocol):
    """Protocol for checking similarity between questions."""
    
    async def check_similarity(self, new_question: str, cached_questions: List[str]) -> Optional[str]:
        """
        Check if a new question is similar to any cached questions.

        Args:
            new_question (str): The new question to check.
            cached_questions (List[str]): List of previously asked questions.

        Returns:
            Optional[str]: The similar question if found, None otherwise.
        """
        ...

class AvailableModelsConfig(BaseModel):
    """Configuration for available AI models."""
    
    models: List[AvailableModel] = Field(
        default=[
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307"
        ],
        description="List of available AI models"
    )

# Create an instance of the config
AVAILABLE_MODELS_CONFIG = AvailableModelsConfig()

# Use the validated models list
AVAILABLE_MODELS = AVAILABLE_MODELS_CONFIG.models

class BaseAgentConfig(BaseModel):
    """Base configuration for AI agents."""
    
    max_retries: int = Field(default=3, ge=1, description="Maximum number of API call retries")
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries in seconds")

class APIConfig(BaseModel):
    """Configuration for API calls."""
    
    model: str = Field(default="claude-3-5-sonnet-20240620", description="AI model to use")
    temperature: float = Field(default=0.0, ge=0, le=1, description="Temperature setting for AI responses")

    @property
    def max_tokens(self) -> int:
        """
        Get the maximum number of tokens for the selected model.

        Returns:
            int: Maximum number of tokens.
        """
        MODEL_MAX_TOKENS = {
            "claude-3-opus-20240229": 4000,
            "claude-3-5-sonnet-20240620": 8000,
            "claude-3-haiku-20240307": 4000
        }
        return MODEL_MAX_TOKENS.get(self.model, 4000)

class RateLimiter:
    """Implements rate limiting for API calls."""
    def __init__(self, rate_limit: float):
        """
        Initialize the RateLimiter.
        Args:
        rate_limit (float): Number of calls allowed per second.
        """
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.updated_at = time.time()

    async def acquire(self):
        """Acquire a slot for an API call."""
        now = time.time()
        time_passed = now - self.updated_at
        self.tokens = min(self.rate_limit, self.tokens + time_passed * self.rate_limit)
        self.updated_at = now

        if self.tokens < 1:
            await asyncio.sleep((1 - self.tokens) / self.rate_limit)
        self.tokens -= 1

def rate_limited(func):
    """
    Decorator to apply rate limiting to a function.
    Args:
    func: The function to be rate limited.
    Returns:
    A wrapper function that applies rate limiting.
    """
    limiter = RateLimiter(API_RATE_LIMIT)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        await limiter.acquire()
        return await func(*args, **kwargs)

    return wrapper

class ClarificationQuestion(BaseModel):
    """Model for a clarification question."""
    
    question: str = Field(..., description="The clarification question")
    reason: str = Field(..., description="Reason for asking this question")

class ClarificationQuestions(BaseModel):
    """Model for a list of clarification questions."""
    
    questions: List[ClarificationQuestion] = Field(..., description="List of clarification questions")

class ClarificationResponse(BaseModel):
    """Model for a response to a clarification question."""
    
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The user's answer to the question")

class ClarificationResponses(BaseModel):
    """Model for a list of clarification responses."""
    
    responses: List[ClarificationResponse] = Field(..., description="List of clarification responses")

class SchemaField(BaseModel):
    type: str
    description: Optional[str] = None

class OutputSchema(BaseModel):
    fields: Dict[str, SchemaField]

class GeneratedPrompt(BaseModel):
    system_prompt: str = Field(..., description="The system prompt providing context and guidelines")
    user_prompt: str = Field(..., description="The specific task prompt")
    output_schema: OutputSchema = Field(..., description="Schema for the expected output")

class StructuredPrompt(BaseModel):
    system_prompt: str
    user_prompt: str
    output_schema: OutputSchema
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def format_prompt(self, **kwargs) -> 'StructuredPrompt':
        new_prompt = self.model_copy(deep=True)
        new_prompt.user_prompt = new_prompt.user_prompt.format(**kwargs)
        return new_prompt

    def to_api_parameters(self) -> Dict[str, Any]:
        params = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.user_prompt}
            ],
            "system": self.system_prompt,
            "max_tokens": self.max_tokens if self.max_tokens is not None else 4000
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.stop_sequences:
            params["stop_sequences"] = self.stop_sequences
        return params

    def validate_response(self, response: Dict[str, Any]) -> Any:
        try:
            logger.debug(f"Response data to validate: {response}")
            DynamicModel = create_dynamic_model(self.output_schema)
            validated_data = DynamicModel(**response)
            logger.debug(f"Validated data: {validated_data}")
            return validated_data
        except ValidationError as e:
            logger.error(f"Response validation failed: {e}")
            raise

def create_dynamic_model(schema: OutputSchema) -> Type[BaseModel]:
    fields = {}
    for field_name, field_info in schema.fields.items():
        # Remove angle brackets from the type string
        type_str = field_info.type.strip('<>')
        
        if type_str == 'class \'float\'':
            field_type = float
        elif type_str == 'class \'int\'':
            field_type = int
        elif type_str == 'class \'str\'':
            field_type = str
        elif type_str.startswith('typing.Literal'):
            # Handle Literal types
            literal_values = re.findall(r"'([^']*)'", type_str)
            field_type = Literal[tuple(literal_values)]
        else:
            # For other types, try to evaluate, but use Any as a fallback
            try:
                field_type = eval(type_str)
            except:
                logger.warning(f"Could not evaluate type '{type_str}' for field '{field_name}'. Using Any.")
                field_type = Any

        field_args = {'description': field_info.description} if field_info.description else {}
        fields[field_name] = (field_type, Field(**field_args))
    
    return create_model('DynamicModel', **fields)

class UnifiedPromptManager(BaseModel):
    """Unified manager for creating, storing, and generating structured prompts."""

    client: AsyncAnthropic
    catalog: Dict[str, StructuredPrompt] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def add_to_catalog(self, name: str, prompt: StructuredPrompt):
        """Add a prompt to the catalog."""
        self.catalog[name] = prompt

    def get_from_catalog(self, name: str) -> Optional[StructuredPrompt]:
        """Retrieve a prompt from the catalog."""
        return self.catalog.get(name)

    @staticmethod
    def create_prompts_prompt() -> StructuredPrompt:
        """Create a structured prompt for generating system and task prompts."""
        output_schema = OutputSchema(fields={
            "generated_prompts": SchemaField(type="List[GeneratedPrompt]", description="List of generated prompts")
        })

        return StructuredPrompt(
            system_prompt="""
            You are an expert prompt engineer working for Anthropic. Your task is to create system prompts and user prompts for helper agents based on given specifications.
            Your response must be a valid JSON object that conforms to the Pydantic model structure provided in the output_schema field.
            Each prompt you generate should include a system prompt, a user prompt, and an output schema that defines the expected structure of the agent's response.
            """,
            user_prompt="""
            Create {num_prompts} prompt(s) for helper agents with the following details:
            Task: {task}
            Domain: {domain}
            Required Expertise: {required_expertise}

            For each prompt, ensure:
            1. The system prompt provides context and guidelines for the helper agent's role and capabilities.
            2. The user prompt clearly articulates the specific task to be performed.
            3. The output schema defines the expected structure of the agent's response.

            Format your response as a JSON object with the following structure:
            {{
                "generated_prompts": [
                    {{
                        "system_prompt": "...",
                        "user_prompt": "...",
                        "output_schema": {{
                            "fields": {{
                                "field_name": {{"type": "...", "description": "..."}},
                                ...
                            }}
                        }}
                    }},
                    ...
                ]
            }}

            Ensure that each prompt instructs the helper agent to provide its response as a structured JSON object that can be validated against the provided output schema.
            """,
            output_schema=output_schema,
            model="claude-3-5-sonnet-20240620",
            temperature=0.4,
            metadata={
                "purpose": "Generate system and task prompts for helper agents",
                "version": "1.1"
            }
        )

    async def generate_prompts(self, task: str, domain: str, required_expertise: str, num_prompts: int = 1) -> List[StructuredPrompt]:
        """Generate structured prompts based on the given specifications."""
        generator_prompt = self.create_prompts_prompt()
        formatted_prompt = generator_prompt.format_prompt(
            task=task,
            domain=domain,
            required_expertise=required_expertise,
            num_prompts=num_prompts
        )

        try:
            response = await self.client.messages.create(**formatted_prompt.to_api_parameters())
            result = formatted_prompt.validate_response(response.content[0].text)
            
            generated_prompts = []
            for prompt_data in result.generated_prompts:
                enhanced_prompt = StructuredPrompt(
                    **prompt_data.model_dump(),
                    model="claude-3-5-sonnet-20240620",
                    temperature=0.4
                )
                generated_prompts.append(enhanced_prompt)
            
            return generated_prompts
        except Exception as e:
            logger.error(f"Failed to generate prompts: {e}")
            return []

    async def execute_prompt(self, prompt: StructuredPrompt, **kwargs) -> Any:
        """Execute a prompt and return the validated result."""
        formatted_prompt = prompt.format_prompt(**kwargs)
        response = await self.client.messages.create(**formatted_prompt.to_api_parameters())
        return formatted_prompt.validate_response(response.content[0].text)

class QuestionCacheEntry(BaseModel):
    """Model for a cached question and its answer."""

    question: str = Field(..., description="The cached question")
    answer: str = Field(..., description="The cached answer")

class GlobalQuestionCache(BaseModel):
    """Global cache for questions and answers."""

    cache: Dict[str, QuestionCacheEntry] = Field(default_factory=dict, description="Dictionary of cached questions and answers")

    def add(self, question: str, answer: str):
        """
        Add a question and its answer to the cache.

        Args:
            question (str): The question to cache.
            answer (str): The answer to cache.
        """
        self.cache[question] = QuestionCacheEntry(question=question, answer=answer)
        logger.debug(f"Added question to cache: {question}")

    def get(self, question: str) -> Optional[str]:
        """
        Get the cached answer for a question.

        Args:
            question (str): The question to look up.

        Returns:
            Optional[str]: The cached answer if found, None otherwise.
        """
        entry = self.cache.get(question)
        if entry:
            logger.debug(f"Cache hit for question: {question}")
            return entry.answer
        logger.debug(f"Cache miss for question: {question}")
        return None

    def get_all(self) -> Dict[str, str]:
        """
        Get all cached questions and answers.

        Returns:
            Dict[str, str]: Dictionary of all cached questions and answers.
        """
        return {k: v.answer for k, v in self.cache.items()}

    def get_all_questions(self) -> List[str]:
        """
        Get all cached questions.

        Returns:
            List[str]: List of all cached questions.
        """
        return list(self.cache.keys())

class SimilarityCheckerConfig(BaseModel):
    """Configuration for the similarity checker."""

    model: str = Field(default="claude-3-5-sonnet-20240620", description="AI model to use for similarity checking")
    max_tokens: int = Field(default=2000, ge=500, le=4000, description="Maximum number of tokens for similarity checking")
    temperature: float = Field(default=0.0, ge=0, le=1, description="Temperature setting for similarity checking")
    rate_limit_calls: int = Field(default=5, ge=0, le=10, description="Number of API calls allowed per period")
    rate_limit_period: float = Field(default=1.0, ge=1.0, le=5.0, description="Time period for rate limiting in seconds")

class SimilarityChecker(BaseModel):
    """Checker for finding similar questions."""

    client: AsyncAnthropic
    config: SimilarityCheckerConfig = Field(default_factory=SimilarityCheckerConfig)
    rate_limiter: RateLimiter = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the SimilarityChecker with rate limiting."""
        super().__init__(**data)
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_calls, 
            self.config.rate_limit_period
        )

    @rate_limited
    async def check_similarity(self, new_question: str, cached_questions: List[str]) -> Optional[str]:
        """
        Check if a new question is similar to any cached questions.

        Args:
            new_question (str): The new question to check.
            cached_questions (List[str]): List of previously asked questions.

        Returns:
            Optional[str]: The similar question if found, None otherwise.
        """
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
            if "No similar question found" not in result:
                logger.info(f"Similar question found: {result}")
                return result
            logger.info("No similar question found")
            return None
        except Exception as e:
            logger.error(f"Error in check_similarity: {str(e)}")
            return None

class UserInteractionManager(BaseModel):
    """Manager for user interactions and caching."""

    global_cache: GlobalQuestionCache
    similarity_checker: SimilarityChecker

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def get_user_input(self, prompt: str) -> str:
        """
        Get user input, checking the cache for similar questions first.

        Args:
            prompt (str): The prompt to display to the user.

        Returns:
            str: The user's response or the cached response if found.
        """
        similar_question = await self.similarity_checker.check_similarity(
            prompt, 
            self.global_cache.get_all_questions()
        )
        
        if similar_question:
            cached_response = self.global_cache.get(similar_question)
            logger.info(f"Found similar question: '{similar_question}'")
            logger.info(f"Using cached response: '{cached_response}'")
            return cached_response
        
        response = input(prompt + "\nYour response: ")
        self.global_cache.add(prompt, response)
        logger.debug(f"Added new question and response to cache: {prompt}")
        return response

class BaseAgent(BaseModel):
    client: AsyncAnthropic
    config: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, client: AsyncAnthropic, **config):
        return cls(client=client, config=config)
    
    def create_structured_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Union[Type[BaseModel], Dict[str, Any], OutputSchema],
        model: AvailableModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StructuredPrompt:
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            output_schema = OutputSchema(fields={
                field: SchemaField(
                    type=str(field_info.annotation),
                    description=field_info.description
                )
                for field, field_info in output_schema.model_fields.items()
            })
        elif isinstance(output_schema, dict):
            output_schema = OutputSchema(fields=output_schema)
        elif not isinstance(output_schema, OutputSchema):
            raise ValueError(f"Invalid output_schema type: {type(output_schema)}")

        schema_description = self.schema_to_string(output_schema)
        user_prompt += f"\n\nPlease provide your response as a valid JSON object that conforms to the following schema:\n{schema_description}\n\nEnsure your response can be parsed directly by the specified Pydantic model."

        return StructuredPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            metadata=metadata or {}
        )

    def schema_to_string(self, schema: OutputSchema) -> str:
        result = "{\n"
        for field_name, field_info in schema.fields.items():
            result += f'  "{field_name}": <{field_info.type}>'
            if field_info.description:
                result += f' // {field_info.description}'
            result += ',\n'
        result = result.rstrip(',\n') + "\n}"
        return result

    async def call_api(self, structured_prompt: StructuredPrompt) -> Any:
        api_params = structured_prompt.to_api_parameters()

        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 1.0)

        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(**api_params)
                raw_response = response.content[0].text
                logger.info(f"API call successful on attempt {attempt + 1}")
                logger.debug(f"Raw API response: {raw_response}")
                
                # Extract JSON from the response
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    logger.debug(f"Extracted JSON string: {json_str}")
                    try:
                        json_response = json.loads(json_str)
                        logger.debug(f"Parsed JSON response: {json_response}")
                        
                        # Check if the parsed JSON is a dictionary
                        if not isinstance(json_response, dict):
                            raise ValueError(f"Parsed JSON is not a dictionary. Type: {type(json_response)}")
                        
                        return structured_prompt.validate_response(json_response)
                    except json.JSONDecodeError as json_error:
                        logger.error(f"JSON parsing error: {json_error}")
                        raise
                else:
                    logger.error("No JSON object found in the response")
                    raise ValueError("No JSON object found in the response")

            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                logger.error(f"Response validation failed: {e}")
                logger.error(f"Raw response: {raw_response}")
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds.")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in call_api: {str(e)}")
                logger.error(traceback.format_exc())
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds.")
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def ask_for_clarification(self, topic: str, context: str, user_input_handler: UserInputProtocol) -> ClarificationResponses:
        clarification_prompt = self.create_structured_prompt(
            system_prompt="""
            You are an AI assistant helping to create a business plan. Your task is to identify areas in the given context that need clarification or more information from the user.
            Generate specific questions that will help gather the necessary details.
            """,
            user_prompt=f"""
            Topic: {topic}
            Context: {context}

            Based on this information, what specific questions should we ask the user to gather more details or clarify any points?
            Provide your response as a JSON object that conforms to the following Pydantic model:

            class ClarificationQuestions(BaseModel):
                questions: List[ClarificationQuestion]

            class ClarificationQuestion(BaseModel):
                question: str
                reason: str

            Limit your response to the most critical questions (maximum 5).
            Ensure your response can be parsed directly into these Pydantic models.
            """,
            output_schema=ClarificationQuestions,
            model="claude-3-5-sonnet-20240620",
            temperature=0.4,
            metadata={"purpose": "Generate clarification questions"}
        )

        try:
            clarification_questions = await self.call_api(clarification_prompt)
            logger.debug(f"Generated clarification questions: {clarification_questions}")
        except Exception as e:
            logger.error(f"Failed to generate clarification questions: {e}")
            logger.error(traceback.format_exc())
            return ClarificationResponses(responses=[])

        user_responses = []
        for question in clarification_questions.questions:
            logger.info(f"Asking clarification question: {question.question}")
            logger.debug(f"Reason for asking: {question.reason}")
            user_response = await user_input_handler.get_user_input(question.question)
            user_responses.append(ClarificationResponse(question=question.question, answer=user_response))

        return ClarificationResponses(responses=user_responses)

class PromptAgent(BaseAgent):
    """Agent for creating prompts."""

    async def create_prompts(self, domain: str, task: str, required_expertise: str) -> Dict[str, str]:
        """
        Create system and task prompts for a given domain and task.

        Args:
            domain (str): The domain of the task.
            task (str): The task description.
            required_expertise (str): The required expertise for the task.

        Returns:
            Dict[str, str]: Dictionary containing system_prompt and task_prompt.
        """
        class PromptOutput(BaseModel):
            system_prompt: str
            task_prompt: str

        structured_prompt = self.create_structured_prompt(
            system_prompt="""
            You are an expert prompt engineer working for Anthropic. Create a system prompt and a task prompt for a helper agent based on the given domain, task, and required expertise.
            Your response must be a valid JSON object that conforms to the Pydantic model structure provided in the output_model field.
            The system prompt should provide context and guidelines for the helper agent's role and capabilities.
            The task prompt should clearly articulate the specific task to be performed and instruct the agent to respond in a structured JSON format.
            """,
            user_prompt=f"""
            Create prompts for a helper agent with the following details:
            Domain: {domain}
            Task: {task}
            Required Expertise: {required_expertise}
            Ensure that the task prompt explicitly instructs the helper agent to provide its response as a structured JSON object with clearly defined keys relevant to the task.
            """,
            output_schema=PromptOutput,
            model="claude-3-5-sonnet-20240620",
            temperature=0.4,
            metadata={
                "purpose": "Generate system and task prompts for helper agents",
                "version": "1.0"
            }
        )
        
        logger.debug(f"PromptAgent - Input: Domain: {domain}, Task: {task}, Expertise: {required_expertise}")
        
        try:
            result = await self.call_api(structured_prompt)
            logger.info("Successfully created prompts")
            logger.debug(f"PromptAgent - Created Prompts: {json.dumps(result.model_dump(), indent=2)}")
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error creating prompts: {e}. Using default prompts.")
            return {
                "system_prompt": f"You are an expert in {domain} with expertise in {required_expertise}. Provide detailed and structured responses.",
                "task_prompt": f"Complete the following task: {task}. Provide your response in a structured JSON format."
            }
        
class TaskAssessment(BaseModel):
    complexity: float = Field(..., ge=0, le=1, description="Estimated task complexity (0-1)")
    recommended_model: AvailableModel = Field(..., description="Recommended AI model for the task")
    max_tokens: int = Field(..., gt=0, description="Recommended max tokens for the task")
    temperature: float = Field(..., ge=0, le=1, description="Recommended temperature for the task")
    estimated_steps: int = Field(..., gt=0, description="Estimated number of steps to complete the task")

class PlanStep(BaseModel):
    step_number: int
    description: str
    details: str
    assigned_model: AvailableModel

class Plan(BaseModel):
    plan: List[PlanStep]

class PlannerAgent(BaseAgent):
    async def assess_task(self, task: str) -> TaskAssessment:
        assessment_prompt = self.create_structured_prompt(
            system_prompt="""
            You are an expert AI system analyst. Your task is to assess the given task and provide recommendations for its execution.
            Consider the complexity of the task, the most suitable AI model, and appropriate parameters for execution.
            Your assessment should be thorough and take into account various factors such as the task's scope, potential challenges, and required expertise.
            """,
            user_prompt=f"""
            Please assess the following task and provide recommendations:

            Task: {task}

            Provide your assessment as a JSON object that conforms to the specified output schema.
            For the 'recommended_model' field, choose from the following options:
            {", ".join(AvailableModel.__args__)}
            """,
            output_schema=TaskAssessment,
            model="claude-3-opus-20240229",
            temperature=0.4
        )

        try:
            assessment = await self.call_api(assessment_prompt)
            logger.info(f"Task assessment: {assessment}")
            return assessment
        except Exception as e:
            logger.error(f"Failed to assess task: {e}")
            logger.error(traceback.format_exc())
            return TaskAssessment(
                complexity=0.5,
                recommended_model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                temperature=0.5,
                estimated_steps=3
            )

    async def create_plan(self, task: str) -> Plan:
        assessment = await self.assess_task(task)
        logger.debug(f"Task assessment for plan creation: {assessment}")

        plan_prompt = self.create_structured_prompt(
            system_prompt="""
            You are a planning expert. Create a structured plan for the given task.
            Your response must be a valid JSON object that conforms to the output model structure.
            For the "assigned_model" field, choose from the following models:
            claude-3-opus-20240229, claude-3-5-sonnet-20240620, claude-3-haiku-20240307
            Assign models strategically, considering that Opus is most capable but slowest, Haiku is fastest but least capable, and Sonnet is a balance between the two that also incorporates caching.
            """,
            user_prompt=f"""
            Create a detailed plan for the following task: {task}

            Task Assessment:
            - Complexity: {assessment.complexity}
            - Recommended Model: {assessment.recommended_model}
            - Estimated Steps: {assessment.estimated_steps}

            Ensure your plan takes into account the assessed complexity and estimated number of steps.
            Provide your plan as a JSON object that conforms to the specified output schema.
            """,
            output_schema=Plan,
            model=assessment.recommended_model,
            max_tokens=assessment.max_tokens,
            temperature=assessment.temperature
        )

        try:
            result = await self.call_api(plan_prompt)
            logger.info(f"Created plan: {result}")
            return Plan(**result.dict())  # Convert DynamicModel to dict before passing to Plan
        except Exception as e:
            logger.error(f"Failed to create a valid plan: {e}")
            logger.error(traceback.format_exc())
            return Plan(plan=[PlanStep(
                step_number=1,
                description="Execute the task",
                details="Complete the entire task in one step",
                assigned_model="claude-3-5-sonnet-20240620"
            )])

    async def refine_plan(self, initial_plan: Plan, task: str, assessment: TaskAssessment) -> Plan:
        refine_prompt = self.create_structured_prompt(
            system_prompt="""
            You are a planning expert. Review and refine the given plan to make it more comprehensive.
            Ensure the plan meets or exceeds the estimated number of steps and covers all aspects of the task in detail.
            Your response must be a valid JSON object that conforms to the output model structure.
            """,
            user_prompt=f"""
            Original task: {task}
            
            Initial plan:
            {json.dumps(initial_plan.model_dump(), indent=2)}
            
            Task Assessment:
            - Complexity: {assessment.complexity}
            - Recommended Model: {assessment.recommended_model}
            - Estimated Steps: {assessment.estimated_steps}

            Please refine this plan to make it more comprehensive and detailed.
            Ensure there are at least {assessment.estimated_steps} steps and that all aspects of the task are covered.
            Provide your refined plan as a JSON object that conforms to the specified output schema.
            """,
            output_schema=Plan,
            model=assessment.recommended_model,
            max_tokens=assessment.max_tokens,
            temperature=assessment.temperature
        )

        try:
            refined_result = await self.call_api(refine_prompt)
            logger.info(f"Refined plan: {refined_result}")
            return refined_result
        except Exception as e:
            logger.error(f"Failed to refine plan: {e}")
            logger.error(traceback.format_exc())
            return initial_plan
        
class StepResult(BaseModel):
    step_number: int
    action_taken: str
    outcome: str
    success: bool
    challenges: List[str]
    next_steps: List[str]
    

class ExecutorAgent(BaseAgent):
    async def execute_step(self, plan: Plan) -> List[StepResult]:
        results = []
        for step in plan.plan:
            result = await self._execute_single_step(step)
            results.append(result)
        return results

    async def execute_plan(self, plan: Plan) -> List[StepResult]:
        results = []
        for step in plan.plan:
            result = await self._execute_single_step(step)
            results.append(result)
        return results

    async def _execute_single_step(self, step: PlanStep) -> StepResult:
        structured_prompt = self.create_structured_prompt(
            system_prompt=f"""
            You are an AI assistant specialized in executing various tasks.
            Your current task is to execute the following step:
            
            Step Number: {step.step_number}
            Description: {step.description}
            Details: {step.details}

            Execute this step to the best of your abilities, considering the given context and objectives.
            Provide a detailed report of your actions, outcomes, challenges faced, and suggested next steps.
            """,
            user_prompt="""
            Please execute the step described above and provide a comprehensive report.
            Include specific actions taken, outcomes achieved, any challenges encountered, and recommended next steps.
            If relevant, include any additional data or insights gained during the execution of this step.

            Ensure your response is detailed, professional, and directly relevant to the task at hand.
            """,
            output_schema=StepResult,
            model=step.assigned_model,
            temperature=0.2
        )

        try:
            result = await self.call_api(structured_prompt)
            logger.info(f"Executed step {step.step_number}: {result.action_taken}")
            return result
        except Exception as e:
            logger.error(f"Failed to execute step {step.step_number}: {e}")
            logger.error(traceback.format_exc())
            return StepResult(
                step_number=step.step_number,
                action_taken="Failed to execute step",
                outcome="Execution failed due to an error",
                success=False,
                challenges=[f"Error during execution: {str(e)}"],
                next_steps=["Retry execution with simplified input", "Review and adjust the step if necessary"]
            )

    async def _attempt_recovery(self, failed_step: PlanStep, failed_result: StepResult) -> StepResult:
        recovery_prompt = self.create_structured_prompt(
            system_prompt=f"""
            You are an AI assistant specialized in problem-solving and error recovery.
            A step in the task execution process has failed.
            Your task is to analyze the failure and attempt to recover or provide an alternative approach.

            Failed Step: {json.dumps(failed_step.model_dump(), indent=2)}
            Failure Details: {json.dumps(failed_result.model_dump(), indent=2)}

            Propose a recovery strategy or alternative approach to achieve the step's objective.
            """,
            user_prompt="""
            Based on the information about the failed step and the failure details, please:
            1. Analyze the root cause of the failure
            2. Propose a recovery strategy or alternative approach
            3. If possible, execute the recovery strategy
            4. Report on the results of the recovery attempt

            Provide your response as a StepResult object, indicating whether the recovery was successful and detailing the actions taken.
            """,
            output_schema=StepResult,
            model=failed_step.assigned_model,
            temperature=0.3
        )

        try:
            recovery_result = await self.call_api(recovery_prompt)
            logger.info(f"Recovery attempt for step {failed_step.step_number}: {recovery_result.action_taken}")
            return recovery_result
        except Exception as e:
            logger.error(f"Failed to execute recovery for step {failed_step.step_number}: {e}")
            return StepResult(
                step_number=failed_step.step_number,
                action_taken="Failed to execute recovery",
                outcome="Recovery attempt failed",
                success=False,
                challenges=[f"Error during recovery: {str(e)}"],
                next_steps=["Manual intervention required", "Reassess the overall plan"],
                additional_data={"error_details": str(e), "original_failure": failed_result.model_dump()}
            )

    async def _create_transition(self, current_result: StepResult, next_step: PlanStep) -> StepResult:
        transition_prompt = self.create_structured_prompt(
            system_prompt=f"""
            You are an AI assistant specializing in creating smooth transitions between task steps.
            Your task is to create a transition between two steps in the execution process.

            Current Step Result: {json.dumps(current_result.model_dump(), indent=2)}
            Next Step: {json.dumps(next_step.model_dump(), indent=2)}

            Create a transition that summarizes the current progress and sets the stage for the next step.
            """,
            user_prompt="""
            Please create a transition between the current step and the next step. Your transition should:
            1. Summarize the key outcomes of the current step
            2. Identify any important insights or data to carry forward
            3. Introduce the objectives of the next step
            4. Explain how the next step builds upon or relates to the work done so far

            Provide your response as a StepResult object, treating this transition as a step in itself.
            """,
            output_schema=StepResult,
            model=next_step.assigned_model,
            temperature=0.2
        )

        try:
            transition_result = await self.call_api(transition_prompt)
            logger.info(f"Created transition to step {next_step.step_number}: {transition_result.action_taken}")
            return transition_result
        except Exception as e:
            logger.error(f"Failed to create transition to step {next_step.step_number}: {e}")
            return StepResult(
                step_number=current_result.step_number + 0.5,
                action_taken="Failed to create transition",
                outcome="Transition creation failed",
                success=False,
                challenges=[f"Error during transition creation: {str(e)}"],
                next_steps=["Proceed to next step without explicit transition", "Review overall plan coherence"],
                additional_data={"error_details": str(e)}
            )

    async def analyze_execution_results(self, results: List[StepResult]) -> Dict[str, Any]:
        analysis_prompt = self.create_structured_prompt(
            system_prompt="""
            You are an AI assistant specializing in process analysis and improvement.
            Your task is to analyze the execution results of a multi-step task.

            Provide a comprehensive analysis of the execution process, including:
            1. Overall success rate
            2. Key achievements
            3. Major challenges encountered
            4. Areas for improvement
            5. Recommendations for future iterations
            """,
            user_prompt=f"""
            Please analyze the following execution results and provide your insights:

            {json.dumps([result.model_dump() for result in results], indent=2)}

            Your analysis should be thorough and provide actionable insights for improving the execution process.
            """,
            output_schema={
                "overall_success_rate": float,
                "key_achievements": List[str],
                "major_challenges": List[str],
                "areas_for_improvement": List[str],
                "recommendations": List[str],
                "additional_insights": Dict[str, Any]
            },
            model="claude-3-opus-20240229",
            temperature=0.3
        )

        try:
            analysis_result = await self.call_api(analysis_prompt)
            logger.info("Completed execution results analysis")
            return analysis_result
        except Exception as e:
            logger.error(f"Failed to analyze execution results: {e}")
            return {
                "overall_success_rate": 0.0,
                "key_achievements": [],
                "major_challenges": ["Failed to complete analysis"],
                "areas_for_improvement": ["Execution result analysis process"],
                "recommendations": ["Review and improve the analysis capability of the ExecutorAgent"],
                "additional_insights": {"error_details": str(e)}
            }

class ReportSection(BaseModel):
    title: str
    content: str

class DynamicReport(BaseModel):
    sections: List[ReportSection]

class ReportStructure(BaseModel):
    sections: List[str]

class PydanticJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)

class SynthesizerAgent(BaseAgent):
    async def determine_report_structure(self, task: str, execution_results: List[Dict[str, Any]]) -> ReportStructure:
        structure_prompt = self.create_structured_prompt(
            system_prompt="""
            You are an expert report structuring AI. Your task is to determine the most appropriate
            structure for a synthesis report based on the given task and execution results.
            Consider the nature of the task, the complexity of the results, and the most effective
            way to present the information to the intended audience.
            """,
            user_prompt=f"""
            Task: {task}

            Execution Results Summary:
            {json.dumps(execution_results, indent=2)}

            Based on this information, propose an appropriate structure for the synthesis report.
            The structure should be a list of section titles that will effectively communicate
            the results, insights, and recommendations derived from the task execution.

            Provide your response as a JSON object that conforms to the specified output schema.
            """,
            output_schema=ReportStructure,
            model="claude-3-5-sonnet-20240620",
            temperature=0.4
        )

        try:
            structure = await self.call_api(structure_prompt)
            logger.info(f"Determined report structure: {structure.sections}")
            return structure
        except Exception as e:
            logger.error(f"Failed to determine report structure: {e}")
            return ReportStructure(sections=["Executive Summary", "Introduction", "Methodology", "Results", "Conclusion"])

    async def synthesize(self, task: str, execution_results: List[Dict[str, Any]], output_file: str) -> str:
        execution_results = [
            result.dict() if hasattr(result, 'dict') else result
            for result in execution_results
        ]

        report_structure = await self.determine_report_structure(task, execution_results)

        synthesis_prompt = self.create_structured_prompt(
            system_prompt=f"""
            You are an expert at synthesizing information and writing comprehensive reports.
            Analyze the results from multiple executed steps and provide a detailed, well-structured report.
            Your report should be clear, concise, and suitable for a professional audience.
            Structure your report according to the provided sections, ensuring that each section
            contains relevant and insightful information derived from the task execution results.
            """,
            user_prompt=f"""
            Task: {task}

            Execution Results:
            {json.dumps(execution_results, indent=2)}

            Generate a comprehensive synthesis report using the following structure:
            {json.dumps(report_structure.dict(), indent=2)}

            Ensure that each section provides valuable insights, analysis, or recommendations
            based on the task execution results. Your response should be a valid JSON object
            conforming to the output schema, with each section's content written in Markdown format.

            Provide your synthesis report as a JSON object that conforms to the specified output schema.
            """,
            output_schema=DynamicReport,
            model="claude-3-5-sonnet-20240620",
            max_tokens=8000,
            temperature=0.2
        )

        try:
            report = await self.call_api(synthesis_prompt)
            
            markdown_content = ""
            for section in report.sections:
                if isinstance(section, dict):
                    section = ReportSection(**section)
                markdown_content += f"# {section.title}\n\n{section.content}\n\n"
            
            # Ensure we have a valid output file path
            if not output_file or not output_file.strip():
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"business_plan_{timestamp}.md"
            
            # Ensure the output file has a .md extension
            if not output_file.lower().endswith('.md'):
                output_file += '.md'
            
            # Create the directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Write the content to the file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Synthesis report written to {output_file}")
            return output_file
        except Exception as e:
            error_message = f"Failed to generate or write synthesis report: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            
            # Create an error report file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = f"synthesis_error_{timestamp}.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Error occurred while generating synthesis report:\n{error_message}\n\nTraceback:\n{traceback.format_exc()}")
            
            return error_file


class AgentConfig(BaseModel):
    agent_class: Type[BaseAgent]
    config: Dict[str, Any] = Field(default_factory=dict)

    def create_agent(self, client: AsyncAnthropic) -> BaseAgent:
        return self.agent_class.create(client, **self.config)

class WorkflowStep(BaseModel):
    agent_name: str
    method_name: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None  # New field for conditional execution

class Workflow(BaseModel):
    steps: List[WorkflowStep]

class OrchestratorConfig(BaseModel):
    api_key: str
    agents: Dict[str, AgentConfig]
    default_workflow: Workflow
    default_system_prompt: str = "You are an AI assistant helping with various tasks."

def safe_get(d, key):
    return d.get(key, None)

class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.client = AsyncAnthropic(api_key=config.api_key)
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        for agent_name, agent_config in self.config.agents.items():
            self.agents[agent_name] = agent_config.create_agent(self.client)

    async def execute_workflow(self, workflow: Workflow, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_context.copy()
        for step in workflow.steps:
            logger.debug(f"Starting step: {step.agent_name}.{step.method_name}")
            logger.debug(f"Step args: {step.args}")
            logger.debug(f"Step kwargs: {step.kwargs}")
            
            if step.condition:
                try:
                    condition_result = eval(step.condition, {"context": context, "safe_get": safe_get})
                    if not condition_result:
                        logger.info(f"Skipping step {step.agent_name}.{step.method_name} due to condition: {step.condition}")
                        continue
                except Exception as e:
                    logger.error(f"Error evaluating condition for step {step.agent_name}.{step.method_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    context[f"{step.agent_name}_{step.method_name}_error"] = f"Condition evaluation failed: {str(e)}"
                    continue

            if step.agent_name not in self.agents:
                logger.error(f"Agent '{step.agent_name}' not found in configured agents.")
                context[f"{step.agent_name}_{step.method_name}_error"] = f"Agent '{step.agent_name}' not found"
                continue
            
            agent = self.agents[step.agent_name]
            method = getattr(agent, step.method_name, None)
            if not method:
                logger.error(f"Method '{step.method_name}' not found in agent '{step.agent_name}'.")
                context[f"{step.agent_name}_{step.method_name}_error"] = f"Method '{step.method_name}' not found"
                continue
            
            args = [context.get(arg, arg) if isinstance(arg, str) else arg for arg in step.args]
            kwargs = {k: context.get(v, v) if isinstance(v, str) else v for k, v in step.kwargs.items()}
            
            try:
                if step.method_name == "execute_plan":
                    # Convert the plan to a Plan object if it's not already
                    if not isinstance(args[0], Plan):
                        args[0] = Plan(**args[0])
                result = await method(*args, **kwargs)
                context[f"{step.agent_name}_{step.method_name}_result"] = result
                logger.info(f"Completed step: {step.agent_name}.{step.method_name}")
            except Exception as e:
                logger.error(f"Error in step {step.agent_name}.{step.method_name}: {str(e)}")
                logger.error(traceback.format_exc())
                context[f"{step.agent_name}_{step.method_name}_error"] = str(e)
        
        return context

    async def execute_task(self, task: str, workflow: Optional[Workflow] = None, output_file: Optional[str] = None, initial_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        workflow = workflow or self.config.default_workflow
        initial_context = {
            "task": task,
            "system_prompt": initial_prompt or self.config.default_system_prompt,
            "output_file": output_file or "",  # Allow empty string, SynthesizerAgent will handle it
            **kwargs
        }
        
        try:
            final_context = await self.execute_workflow(workflow, initial_context)
            if "synthesizer_synthesize_result" in final_context:
                final_context["status"] = "completed"
                final_context["output_file"] = final_context["synthesizer_synthesize_result"]
            else:
                final_context["status"] = "partial_completion"
                final_context["error_message"] = "Some steps failed to complete."
            logger.info(f"Task execution completed with status: {final_context['status']}")
            return final_context
        except Exception as e:
            logger.error(f"An error occurred during task execution: {str(e)}", exc_info=True)
            return {
                "task": task,
                "status": "error",
                "error_message": str(e),
                "output_file": f"error_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            }
        
# Example usage:
async def main():
    load_dotenv(override=True)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    config = OrchestratorConfig(
        api_key=api_key,
        agents={
            "planner": AgentConfig(agent_class=PlannerAgent),
            "executor": AgentConfig(agent_class=ExecutorAgent),
            "synthesizer": AgentConfig(agent_class=SynthesizerAgent),
        },
        default_workflow=Workflow(steps=[
            WorkflowStep(agent_name="planner", method_name="assess_task", args=["task"]),
            WorkflowStep(agent_name="planner", method_name="create_plan", args=["task"]),
            WorkflowStep(
                agent_name="executor",
                method_name="execute_plan",
                args=["planner_create_plan_result"],
                condition="safe_get(context, 'planner_create_plan_result') is not None"
            ),
            WorkflowStep(
                agent_name="synthesizer",
                method_name="synthesize",
                args=["task", "executor_execute_plan_result", "output_file"],
                condition="safe_get(context, 'executor_execute_plan_result') is not None"
            ),
        ]),
        default_system_prompt="You are an AI assistant analyzing complex topics."
    )

    orchestrator = Orchestrator(config)
    result = await orchestrator.execute_task(
        "Generate a business plan for a company called MASS (Multi Agent System Solutions) which provides AI solution consulting and an initial product that is an interactive business plan generator.",
        output_file="../../output_files/business_plan.md",
        initial_prompt="You are an expert in business planning. Create a detailed business plan for the context provided.",
    )
    print(json.dumps(result, indent=2,cls=PydanticJSONEncoder))

if __name__ == "__main__":
    asyncio.run(main())