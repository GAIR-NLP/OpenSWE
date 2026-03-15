"""
Interfacing with OpenAI models.
"""

import json
import os
import sys
from typing import Literal, cast

from loguru import logger
from openai import omit, BadRequestError, OpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionFunctionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenaiFunction,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.data_structures import FunctionCallIntent
from app.log import log_and_print
from app.model import common
from app.model.common import Model
import time


class OpenaiModel(Model):
    """
    Base class for creating Singleton instances of OpenAI models.
    We use native API from OpenAI instead of LiteLLM.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        max_output_token: int,
        cost_per_input: float,
        cost_per_output: float,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        # max number of output tokens allowed in model response
        # sometimes we want to set a lower number for models with smaller context window,
        # because output token limit consumes part of the context window
        self.max_output_token = max_output_token
        # client for making request
        self.client: OpenAI | None = None
        self._initialized = True
        self.start_time = time.time()
        self.total_tokens = 0

    def setup(self) -> None:
        """
        Check API key, and initialize OpenAI client.
        """
        if self.client is None:
            key = self.check_api_key()
            self.client = OpenAI(
                api_key=key,
                base_url=os.getenv("OPENAI_API_BASE_URL", None),
                timeout=300,
            )

    def check_api_key(self) -> str:
        key = os.getenv("OPENAI_KEY")
        if not key:
            print("Please set the OPENAI_KEY env var")
            sys.exit(1)
        return key

    def extract_resp_content(
        self, chat_completion_message: ChatCompletionMessage
    ) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_completion_message.content
        if content is None:
            return ""
        else:
            return content

    def extract_resp_func_calls(
        self,
        chat_completion_message: ChatCompletionMessage,
    ) -> list[FunctionCallIntent]:
        """
        Given a chat completion message, extract the function calls from it.
        Args:
            chat_completion_message (ChatCompletionMessage): The chat completion message.
        Returns:
            List[FunctionCallIntent]: A list of function calls.
        """
        result = []
        tool_calls: list[ChatCompletionMessageToolCall] | None = (
            chat_completion_message.tool_calls
        )  # type: ignore
        if tool_calls is None:
            return result

        call: ChatCompletionMessageToolCall
        for call in tool_calls:
            called_func: OpenaiFunction = call.function
            func_name = called_func.name
            func_args_str = called_func.arguments
            # maps from arg name to arg value
            if func_args_str == "":
                args_dict = {}
            else:
                try:
                    args_dict = json.loads(func_args_str, strict=False)
                except json.decoder.JSONDecodeError:
                    args_dict = {}
            func_call_intent = FunctionCallIntent(func_name, args_dict, called_func)
            result.append(func_call_intent)

        return result

    # FIXME: the returned type contains OpenAI specific Types, which should be avoided
    @retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(10))
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[
        str,
        list[ChatCompletionMessageToolCall] | None,
        list[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        """
        Calls the openai API to generate completions for the given inputs.
        Assumption: we only retrieve one choice from the API response.

        Args:
            messages (List): A list of messages.
                            Each item is a dict (e.g. {"role": "user", "content": "Hello, world!"})
            top_p (float): The top_p to use. We usually do not vary this, so not setting it as a cmd-line argument. (from 0 to 1)
            tools (List, optional): A list of tools.

        Returns:
            Raw response and parsed components.
            The raw response is to be sent back as part of the message history.
        """
        if temperature is None:
            temperature = common.MODEL_TEMP

        assert self.client is not None
        if "qwen" in self.name.lower():
            # simple rate-limit guard: ~40k tokens/minute
            elapsed = time.time() - self.start_time
            if elapsed > 0 and (self.total_tokens * 60) > (40000 * elapsed):
                waiting_time = (self.total_tokens * 60 / 40000) - elapsed
                if waiting_time > 0:
                    logger.info(f"waiting for {waiting_time} seconds!")
                    time.sleep(waiting_time)

        try:
            if tools is not None and len(tools) == 1:
                # there is only one tool => force the model to use it
                tool_name = tools[0]["function"]["name"]
                tool_choice = {"type": "function", "function": {"name": tool_name}}
                response = self.client.chat.completions.create(
                    # model=self.name,
                    model=(
                        self.name
                        if not self.name.startswith("litellm-")
                        else self.name[len("litellm-") :]
                    ),
                    messages=messages,  # type: ignore
                    tools=tools,  # type: ignore
                    tool_choice=cast(ChatCompletionToolChoiceOptionParam, tool_choice),
                    temperature=(temperature if self.name.startswith("o1") else omit),
                    response_format=cast(ResponseFormat, {"type": response_format}),
                    max_tokens=(
                        self.max_output_token
                        if not self.name.startswith("o1")
                        else omit
                    ),
                    max_completion_tokens=(
                        self.max_output_token if self.name.startswith("o1") else omit
                    ),
                    top_p=top_p,
                    stream=False,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,  # type: ignore
                    tools=tools if tools is not None else omit,  # type: ignore
                    temperature=(temperature if self.name.startswith("o1") else omit),
                    response_format=cast(ResponseFormat, {"type": response_format}),
                    max_tokens=(
                        self.max_output_token
                        if not self.name.startswith("o1")
                        else omit
                    ),
                    max_completion_tokens=(
                        self.max_output_token if self.name.startswith("o1") else omit
                    ),
                    top_p=top_p,
                    stream=False,
                )

            usage_stats = response.usage
            assert usage_stats is not None

            input_tokens = int(usage_stats.prompt_tokens)
            output_tokens = int(usage_stats.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens
            self.total_tokens += input_tokens
            raw_response = response.choices[0].message
            # log_and_print(f"Raw model response: {raw_response}")
            content = self.extract_resp_content(raw_response)
            raw_tool_calls = raw_response.tool_calls
            func_call_intents = self.extract_resp_func_calls(raw_response)
            return (
                content,
                raw_tool_calls,
                func_call_intents,
                cost,
                input_tokens,
                output_tokens,
            )
        except BadRequestError as e:
            logger.debug("BadRequestError ({}): messages={}", e.code, messages)
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e
        except RateLimitError as e:
            # 捕获 RateLimitError 错误并输出详细信息
            logger.error("RateLimitError occurred: {}", e)
            logger.error("Error Code: {}", e.code)  # 错误代码
            logger.error("Error Message: {}", e.message)  # 错误信息
            # if hasattr(e, 'headers'):
            #     logger.error("Rate Limit Reset Time: %s", e.headers.get('Retry-After', 'Not Provided'))  # 如果有重试时间，也输出
            # if hasattr(e, 'response'):
            #     logger.error("Response content: %s", e.response)  # 输出更多的响应内容

            # 可以选择重新抛出该异常，如果需要进一步处理
            raise e


class Gpt_o1mini(OpenaiModel):
    def __init__(self):
        super().__init__("o1-mini", 8192, 0.000003, 0.000012, parallel_tool_call=True)
        self.note = "Mini version of state of the art. Up to Oct 2023."

    # FIXME: the returned type contains OpenAI specific Types, which should be avoided
    @retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[
        str,
        list[ChatCompletionMessageToolCall] | None,
        list[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        if response_format == "json_object":
            last_content = messages[-1]["content"]
            last_content += "\nYour response MUST start with { and end with }. DO NOT write anything else other than the json. Ignore writing triple-backticks."
            messages[-1]["content"] = last_content
            response_format = "text"

        for msg in messages:
            msg["role"] = "user"
        return super().call(
            messages, top_p, tools, response_format, temperature, **kwargs
        )


class Gpt4o_20240806(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-4o-2024-08-06", 16384, 0.0000025, 0.000010, parallel_tool_call=True
        )
        self.note = "Multimodal model. Up to Apr 2023."


class Gpt4o_20241120(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-4o-2024-11-20", 16384, 0.0000025, 0.000010, parallel_tool_call=True
        )
        self.note = "Multimodal model. Up to Apr 2023."


class Gpt4o_20240513(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-4o-2024-05-13", 4096, 0.000005, 0.000015, parallel_tool_call=True
        )
        self.note = "Multimodal model. Up to Oct 2023."


class Gpt4_Turbo20240409(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-4-turbo-2024-04-09", 4096, 0.00001, 0.00003, parallel_tool_call=True
        )
        self.note = "Turbo with vision. Up to Dec 2023."


class Gpt4_0125Preview(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-4-0125-preview", 4096, 0.00001, 0.00003, parallel_tool_call=True
        )
        self.note = "Turbo. Up to Dec 2023."


class Gpt4_1106Preview(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-4-1106-preview", 4096, 0.00001, 0.00003, parallel_tool_call=True
        )
        self.note = "Turbo. Up to Apr 2023."


class Gpt35_Turbo0125(OpenaiModel):
    # cheapest gpt model
    def __init__(self):
        super().__init__(
            "gpt-3.5-turbo-0125", 1024, 0.0000005, 0.0000015, parallel_tool_call=True
        )
        self.note = "Turbo. Up to Sep 2021."


class Gpt35_Turbo1106(OpenaiModel):
    def __init__(self):
        super().__init__(
            "gpt-3.5-turbo-1106", 1024, 0.000001, 0.000002, parallel_tool_call=True
        )
        self.note = "Turbo. Up to Sep 2021."


class Gpt35_Turbo16k_0613(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-3.5-turbo-16k-0613", 1024, 0.000003, 0.000004)
        self.note = "Turbo. Deprecated. Up to Sep 2021."


class Gpt35_Turbo0613(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-3.5-turbo-0613", 512, 0.0000015, 0.000002)
        self.note = "Turbo. Deprecated. Only 4k window. Up to Sep 2021."


class Gpt4_0613(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-4-0613", 512, 0.00003, 0.00006)
        self.note = "Not turbo. Up to Sep 2021."


class Gpt4o_mini_20240718(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-4o-mini-2024-07-18", 4096, 0.00000015, 0.0000006)


class Gpt4_1_nano(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-4.1-nano", 8192, 0.0000001, 0.0000004)


class Gpt4_1_mini(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-4.1-mini", 8192, 0.0000004, 0.0000016)


class Gpt5_mini(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-5-mini", 8192, 0.00000025, 0.000002)


class Gpt4_1(OpenaiModel):
    def __init__(self):
        super().__init__("gpt-4.1", 8192, 0.000002, 0.000008)


class Gemini_2_5_flash_preview(OpenaiModel):
    def __init__(self):
        super().__init__(
            "google/gemini-2.5-flash-preview", 16384, 0.00000015, 0.0000006
        )


class Gemini_2_5_flash_lite_preview(OpenaiModel):
    def __init__(self):
        super().__init__(
            "google/gemini-2.5-flash-lite-preview-06-17", 8192, 0.00000010, 0.0000004
        )


class Kimi_k2(OpenaiModel):
    def __init__(self):
        super().__init__("moonshotai/kimi-k2", 8192, 0.00000014, 0.00000249)


class Qwen25_72B(OpenaiModel):
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-72B-Instruct-128K", 4096, 0.00000057, 0.00000057)
        self.note = "Qwen2.5-72B."


class Qwen3_235(OpenaiModel):
    def __init__(self):
        # Custom Qwen3-235 via OpenAI-compatible endpoint
        super().__init__("qwen3-235", 4096, 0.0, 0.0)
        self.note = "Custom Qwen3-235."


class DeepSeekV25(OpenaiModel):
    def __init__(self):
        super().__init__("deepseek-chat", 4096, 0.00000014, 0.00000028)
        self.note = "Qwen2.5-72B."


class DeepSeekV3(OpenaiModel):
    def __init__(self):
        super().__init__("deepseek/deepseek-chat-v3-0324", 8192, 0.00000028, 0.00000088)
        # self.note = "Qwen2.5-72B."


class DeepSeek(OpenaiModel):
    def __init__(self):
        super().__init__("deepseek-v3", 4096, 0.00000014, 0.00000028)
        self.note = "Qwen2.5-72B."


# Custom model whose actual name is provided via env var CUSTOM_OPENAI_MODEL_NAME.
# Useful for internal OpenAI-compatible endpoints (e.g., DeepSeek v3.2) with custom model ids.
class CustomOpenAIModel(OpenaiModel):
    def __init__(self):
        model_name = os.getenv("CUSTOM_OPENAI_MODEL_NAME", "deepseek-chat")
        super().__init__(model_name, 4096, 0.0, 0.0)
        self.note = f"Custom OpenAI-compatible model: {model_name}"


class GLM46(OpenaiModel):
    def __init__(self):
        # Custom GLM-4.6 via OpenAI-compatible endpoint
        super().__init__("GLM-4.6", 4096, 0.0, 0.0)
        self.note = "Custom GLM-4.6."


class Claude3_5Sonnet(OpenaiModel):
    def __init__(self):
        super().__init__(
            "claude-3-5-sonnet-20240620",
            8192,
            0.000003,
            0.000015,
            parallel_tool_call=True,
        )
        self.note = "Most intelligent model from Antropic"
        # FIXME: the returned type contains OpenAI specific Types, which should be avoided

    @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[
        str,
        list[ChatCompletionMessageToolCall] | None,
        list[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        if response_format == "json_object":
            last_content = messages[-1]["content"]
            last_content += "\nYour response MUST start with { and end with }. DO NOT write anything else other than the json. Ignore writing triple-backticks. DO NOT start with ```json. Your response MUST start with { and end with }."
            messages[-1]["content"] = last_content
            response_format = "text"

        # for msg in messages:
        #     msg["role"] = "user"
        return super().call(
            messages, top_p, tools, response_format, temperature, **kwargs
        )


class Claude3_7Sonnet(OpenaiModel):
    def __init__(self):
        super().__init__(
            "claude-3-7-sonnet-20250219",
            8192,
            0.000003,
            0.000015,
            parallel_tool_call=True,
        )
        self.note = "Most intelligent model from Antropic"
        # FIXME: the returned type contains OpenAI specific Types, which should be avoided

    @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: list[ChatCompletionFunctionToolParam] | None = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[
        str,
        list[ChatCompletionMessageToolCall] | None,
        list[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        if response_format == "json_object":
            last_content = messages[-1]["content"]
            last_content += "\nWrap the results in a format like ```json<CONTENT>```.."
            messages[-1]["content"] = last_content
            response_format = "text"

        # for msg in messages:
        #     msg["role"] = "user"
        return super().call(
            messages, top_p, tools, response_format, temperature, **kwargs
        )
