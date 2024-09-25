# v4 = v3 + updated orchestrator


import asyncio
import json
import logging
import os
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable,Type,TypeVar,get_type_hints
from abc import ABC, abstractmethod
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError,create_model,field_validator
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
    def __init__(self, rate_limit_calls: int, rate_limit_period: float):
        """
        Initialize the RateLimiter.
        Args:
        rate_limit_calls (int): Number of calls allowed per period.
        rate_limit_period (float): Time period for rate limiting in seconds.
        """
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self.tokens = rate_limit_calls
        self.updated_at = time.time()

    async def acquire(self):
        """Acquire a slot for an API call."""
        now = time.time()
        time_passed = now - self.updated_at
        self.tokens = min(self.rate_limit_calls, self.tokens + time_passed * self.rate_limit_calls / self.rate_limit_period)
        self.updated_at = now

        if self.tokens < 1:
            await asyncio.sleep((1 - self.tokens) * self.rate_limit_period / self.rate_limit_calls)
        self.tokens -= 1

def rate_limited(func):
    """
    Decorator to apply rate limiting to a function.
    Args:
    func: The function to be rate limited.
    Returns:
    A wrapper function that applies rate limiting.
    """
    limiter = RateLimiter(API_RATE_LIMIT, 1.0)  # Assuming a rate limit period of 1 second

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

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: AsyncAnthropic, **data):
        super().__init__(client=client, **data)
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
    user_interaction_manager: UserInteractionManager

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, client: AsyncAnthropic, user_interaction_manager: UserInteractionManager, **config):
        return cls(client=client, user_interaction_manager=user_interaction_manager, config=config)


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

    async def ask_for_clarification(self, topic: str, context: str) -> Dict[str, Any]:
        clarification_prompt = self.create_structured_prompt(
            system_prompt="""
            You are an AI assistant helping to create a business plan. Your task is to identify areas in the given context that need clarification or more information from the user.
            Generate specific questions that will help gather the necessary details, taking into account the actual task and the content of the business plan section.
            """,
            user_prompt=f"""
            Task: {self.user_interaction_manager.global_cache.get('task')}
            Topic: {topic}
            Context: {context}

            Based on the task and the content of the '{topic}' section, what specific questions should we ask the user to gather more details or clarify any points?
            Consider the following:
            1. Key elements that may be missing or unclear in the section
            2. Assumptions that need validation
            3. Industry or market-specific details that could enhance the section
            4. Alignment with the overall task and business objectives

            Provide your response as a JSON object that conforms to the following Pydantic model:

            class ClarificationQuestions(BaseModel):
                questions: List[ClarificationQuestion]

            class ClarificationQuestion(BaseModel):
                question: str
                reason: str

            Limit your response to the most critical questions (maximum 3).
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
            return {}

        user_responses = {}
        for question_dict in clarification_questions.questions:  # Iterate over the list of question dictionaries
            question = question_dict['question']  # Access the 'question' key from the dictionary
            reason = question_dict['reason']  # Access the 'reason' key from the dictionary

            logger.info(f"Asking clarification question: {question}")
            logger.debug(f"Reason for asking: {reason}")
            user_response = await self.user_interaction_manager.get_user_input(question)
            user_responses[question] = user_response

        return user_responses
    
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
    assigned_model: str
    details: str = ""

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
            model="claude-3-5-sonnet-20240620",
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
            initial_plan = await self.call_api(plan_prompt)
            logger.info(f"Created initial plan: {initial_plan}")
            
            # Ask for clarification on each step
            for step in initial_plan.plan:
                clarification = await self.ask_for_clarification(f"Task: {step.description}", json.dumps(step.model_dump()))
                if clarification:
                    # Update the step based on user input
                    update_prompt = self.create_structured_prompt(
                        system_prompt="You are a planning expert. Update the given plan step based on user input.",
                        user_prompt=f"""
                        Original step: {json.dumps(step.model_dump())}
                        User input: {json.dumps(clarification)}

                        Update the step based on this user input. Provide your response as a JSON object that conforms to the PlanStep model.
                        """,
                        output_schema=PlanStep,
                        model=assessment.recommended_model,
                        temperature=0.3
                    )
                    updated_step = await self.call_api(update_prompt)
                    step.description = updated_step.description
                    step.details = updated_step.details

            # Refine the plan after all clarifications
            refined_plan = await self.refine_plan(initial_plan, task, assessment)
            return refined_plan
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
    async def execute_step(self, step: PlanStep) -> Dict[str, Any]:
        structured_prompt = self.create_structured_prompt(
            system_prompt=f"""
            You are an AI assistant specialized in executing various tasks for business plan creation.
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
            Your response should be a valid JSON object that conforms to the StepResult model structure.
            """,
            output_schema=StepResult,
            model=step.assigned_model,
            temperature=0.2
        )

        try:
            result = await self.call_api(structured_prompt)
            logger.info(f"Executed step {step.step_number}: {result.action_taken}")
            return result.model_dump()
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
            ).model_dump()

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
            
            # Get user feedback on each section
            for section_dict in report.sections:  # Iterate over the list of section dictionaries
                title = section_dict['title']  # Access the 'title' key from the dictionary
                content = section_dict['content']  # Access the 'content' key from the dictionary

                feedback = await self.user_interaction_manager.get_user_input(f"Please provide any feedback or suggestions for the '{title}' section:")
                if feedback.strip():
                    # Update the section based on user feedback
                    update_prompt = self.create_structured_prompt(
                        system_prompt="You are an expert report writer. Update the given report section based on user feedback.",
                        user_prompt=f"""
                        Original section:
                        Title: {title}
                        Content: {content}

                        User feedback: {feedback}

                        Update the section based on this feedback. Provide your response as a JSON object that conforms to the ReportSection model.
                        """,
                        output_schema=ReportSection,
                        model="claude-3-5-sonnet-20240620",
                        temperature=0.3
                    )
                    updated_section = await self.call_api(update_prompt)
                    section_dict['title'] = updated_section.title  # Update the 'title' key in the dictionary
                    section_dict['content'] = updated_section.content  # Update the 'content' key in the dictionary
            
            markdown_content = ""
            for section_dict in report.sections:  # Iterate over the list of section dictionaries
                title = section_dict['title']  # Access the 'title' key from the dictionary
                content = section_dict['content']  # Access the 'content' key from the dictionary
                markdown_content += f"# {title}\n\n{content}\n\n"
            
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
            error_file = f"../../output_files/synthesis_error_{timestamp}.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Error occurred while generating synthesis report:\n{error_message}\n\nTraceback:\n{traceback.format_exc()}")
            
            return error_file

class UserPreferences(BaseModel):
    interactivity_level: int = Field(..., ge=1, le=10, description="Level of interactivity, from 1 (lowest) to 10 (highest)")
    output_specificity: int = Field(..., ge=1, le=10, description="Level of output specificity, from 1 (most general) to 10 (most detailed)")

    @field_validator('interactivity_level', 'output_specificity')
    @classmethod
    def validate_level(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"Value must be between 1 and 10, got {v}")
        return v

class Orchestrator(BaseAgent):
    preferences: UserPreferences = Field(...)
    planner: Any = Field(default=None)
    executor: Any = Field(default=None)
    synthesizer: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.planner = PlannerAgent.create(self.client, self.user_interaction_manager)
        self.executor = ExecutorAgent.create(self.client, self.user_interaction_manager)
        self.synthesizer = SynthesizerAgent.create(self.client, self.user_interaction_manager)

    async def orchestrate(self, user_prompt: str) -> str:
        plan = await self.create_plan(user_prompt)
        execution_results = await self.execute_plan(plan)
        output_file = await self.synthesize_results(user_prompt, execution_results)
        return output_file

    async def create_plan(self, user_prompt: str) -> Plan:
        logger.info("Creating plan...")
        try:
            initial_plan = await self.planner.create_plan(user_prompt)
            plan = Plan(plan=[PlanStep(**step) for step in initial_plan.plan])
            
            if self.preferences.interactivity_level > 3:
                plan = await self.get_plan_feedback(plan)
            
            return plan
        except Exception as e:
            logger.error(f"Failed to create a valid plan: {e}")
            logger.error(traceback.format_exc())
            return Plan(plan=[PlanStep(
                step_number=1,
                description="Execute the task",
                details="Complete the entire task in one step",
                assigned_model="claude-3-5-sonnet-20240620"
            )])

    async def get_plan_feedback(self, plan: Plan) -> Plan:
        while True:
            feedback = await self.user_interaction_manager.get_user_input(
                f"Here's the current plan:\n{plan.model_dump_json(indent=2)}\nDo you want to make any changes? (yes/no): "
            )
            if feedback.lower() == 'no':
                return plan
            
            changes = await self.user_interaction_manager.get_user_input(
                "Please describe the changes you'd like to make (e.g., 'Add a step for market research', 'Change step 2 to use a different model'): "
            )
            
            # Apply changes to the plan
            updated_plan = await self.apply_plan_changes(plan, changes)
            plan = updated_plan
        
        return plan

    async def apply_plan_changes(self, plan: Plan, changes: str) -> Plan:
        change_prompt = self.create_structured_prompt(
            system_prompt="You are an AI assistant helping to modify a business plan. Apply the user's requested changes to the given plan.",
            user_prompt=f"""
            Current plan:
            {plan.model_dump_json(indent=2)}

            Requested changes:
            {changes}

            Please apply these changes to the plan. Return the entire updated plan, not just the changes.
            Ensure the output is a valid JSON object that conforms to the Plan model structure.
            """,
            output_schema=Plan,
            model="claude-3-5-sonnet-20240620",
            temperature=0.2
        )

        try:
            updated_plan = await self.call_api(change_prompt)
            logger.info("Plan updated based on user feedback")
            return updated_plan
        except Exception as e:
            logger.error(f"Failed to apply changes to plan: {e}")
            return plan

    async def execute_plan(self, plan: Plan) -> List[Dict[str, Any]]:
        logger.info("Executing plan...")
        results = []
        for step in plan.plan:
            if self.preferences.interactivity_level > 7:
                await self.user_interaction_manager.get_user_input(f"About to execute step {step.step_number}: {step.description}. Press Enter to continue...")
            
            result = await self.executor.execute_step(step)
            results.append(result)
            
            if self.preferences.interactivity_level > 5:
                feedback = await self.user_interaction_manager.get_user_input(f"Step {step.step_number} completed. Any comments or additional instructions? ")
                if feedback.strip():
                    updated_result = await self.incorporate_step_feedback(result, feedback)
                    results[-1] = updated_result
        
        return results

    async def incorporate_step_feedback(self, result: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        feedback_prompt = self.create_structured_prompt(
            system_prompt="You are an AI assistant helping to incorporate user feedback into a completed step of a business plan.",
            user_prompt=f"""
            Original step result:
            {json.dumps(result, indent=2)}

            User feedback:
            {feedback}

            Please incorporate this feedback into the step result. Update the relevant fields of the StepResult model.
            Ensure your response is a valid JSON object that conforms to the StepResult model structure.
            """,
            output_schema=StepResult,
            model="claude-3-5-sonnet-20240620",
            temperature=0.2
        )

        try:
            updated_result = await self.call_api(feedback_prompt)
            logger.info("Step result updated based on user feedback")
            return updated_result.model_dump()
        except Exception as e:
            logger.error(f"Failed to incorporate feedback: {e}")
            return result

    async def synthesize_results(self, task: str, execution_results: List[Dict[str, Any]]) -> str:
        logger.info("Synthesizing results...")
        output_file = f"output_specificity_{self.preferences.output_specificity}.md"
        synthesis = await self.synthesizer.synthesize(task, execution_results, output_file)
        
        if self.preferences.interactivity_level > 3:
            feedback = await self.user_interaction_manager.get_user_input("Review the synthesized results. Any final comments or changes? ")
            if feedback.strip():
                synthesis = await self.incorporate_final_feedback(synthesis, feedback)
        
        return synthesis

    async def incorporate_final_feedback(self, synthesis: str, feedback: str) -> str:
        feedback_prompt = self.create_structured_prompt(
            system_prompt="You are an AI assistant helping to incorporate final user feedback into a business plan.",
            user_prompt=f"""
            Original synthesis:
            {synthesis}

            User feedback:
            {feedback}

            Please incorporate this feedback into the final synthesis. Make any necessary changes or additions.
            Return the entire updated synthesis as a Markdown-formatted string.
            """,
            output_schema=str,
            model="claude-3-5-sonnet-20240620",
            temperature=0.2
        )

        try:
            updated_synthesis = await self.call_api(feedback_prompt)
            logger.info("Final synthesis updated based on user feedback")
            return updated_synthesis
        except Exception as e:
            logger.error(f"Failed to incorporate final feedback: {e}")
            return synthesis


async def main():
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = AsyncAnthropic(api_key=api_key)
    
    # Get user preferences
    while True:
        try:
            interactivity = int(input("Choose interactivity level (1-10, where 1 is lowest and 10 is highest): "))
            specificity = int(input("Choose output specificity (1-10, where 1 is most general and 10 is most detailed): "))
            preferences = UserPreferences(interactivity_level=interactivity, output_specificity=specificity)
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

    user_prompt = input("Enter your task prompt: ")

    # Initialize components
    global_cache = GlobalQuestionCache()
    similarity_checker = SimilarityChecker(client=client)
    user_interaction_manager = UserInteractionManager(global_cache=global_cache, similarity_checker=similarity_checker)

    # Create and run the orchestrator
    orchestrator = Orchestrator(
        client=client,
        user_interaction_manager=user_interaction_manager,
        preferences=preferences
    )
    output_file = await orchestrator.orchestrate(user_prompt)

    print(f"Task completed. Output written to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())