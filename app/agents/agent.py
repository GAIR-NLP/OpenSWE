from abc import ABC, abstractmethod
from app.data_structures import MessageThread, FunctionCallIntent
from app.log import log_exception
from loguru import logger
import os
import json
from collections.abc import Callable, Mapping


class Agent(ABC):
    """
    Abstract base class for all agents.
    Provides per-agent message thread, tool call tracking, and default dispatch_intent.
    """

    api_functions: list[str] = []

    def __init__(self, agent_id):
        # Each agent has its own thread
        self.msg_thread = MessageThread()
        self.agent_id = agent_id
        # Tracking of tool calls
        self.tool_call_sequence: list[dict] = []
        self.tool_call_layers: list[list[dict]] = []
        self.curr_tool: str | None = None
        self.iteration_num = 0
        self.finish_status = True

    def add_user_message(self, text: str):
        """add a user message to the thread"""
        self.msg_thread.add_user(text)

    def add_system_message(self, text: str):
        """add a system message to the thread"""
        self.msg_thread.add_system(text)

    def add_model_message(self, text: str, tools: list):
        """add a model message to the thread"""
        self.msg_thread.add_model(text, tools)

    @abstractmethod
    def run_task(self, print_callback=None, **kwargs) -> tuple[str, str, bool]:
        """
        Execute the agent's primary function.
        Returns:
            - output (str): raw tool or LLM output
            - summary (str): one-line summary
            - success (bool): whether the action succeeded
        """
        pass

    def init_msg_thread(self) -> None:
        pass

    def dispatch_intent(
        self,
        intent: FunctionCallIntent,
        # message_thread: MessageThread = None,
        # print_callback: Callable[[dict], None] | None = None,
    ) -> tuple[str, str, bool]:
        """
        Dispatch a FunctionCallIntent to call the agent's tool methods.
        """

        if intent.func_name not in self.api_functions:
            error = f"Unknown function name {intent.func_name}."
            summary = "You called a tool that does not exist."
            return error, summary, False

        func_obj = getattr(self, intent.func_name)
        try:
            self.curr_tool = intent.func_name
            # If function expects thread
            # if 'message_thread' in func_obj.__code__.co_varnames:
            #     call_res = func_obj(message_thread, print_callback=print_callback)
            # else:
            call_res = func_obj(**intent.arg_values)
        except Exception as e:
            log_exception(e)
            error = str(e)
            summary = "Tool raised an exception."
            call_res = (error, summary, False)

        logger.debug("Result of dispatch_intent: {}", call_res)

        # Record the call
        result, _, ok = call_res
        self.tool_call_sequence.append(
            intent.to_dict_with_result(ok, result, self.agent_id)
        )
        # if not self.tool_call_layers:
        #     self.tool_call_layers.append([])
        # self.tool_call_layers[-1].append(intent.to_dict_with_result(ok,result,self.agent_id))

        return call_res

    def start_new_layer(self):
        self.tool_call_layers.append([])

    def reset_tool_sequence(self):
        self.tool_call_sequence = []

    def dump_tool_sequence(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        seq_file = os.path.join(output_dir, "tool_sequence.json")
        # layer_file = os.path.join(output_dir, 'agent_tool_layers.json')
        with open(seq_file, "w") as f:
            json.dump(self.tool_call_sequence, f, indent=2)
        # with open(layer_file, 'w') as f:
        #     json.dump(self.tool_call_layers, f, indent=2)
