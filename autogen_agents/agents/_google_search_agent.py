from typing import Any, Callable, Dict, List, Optional, Union

from autogen import ConversableAgent
from googleapiclient.discovery import build


class GoogleSearchAgent(ConversableAgent):  # type: ignore[misc]
    """GoogleSearchAgent agent. Search the web for the user and provide the search report.

    `human_input_mode` is default to "NEVER" and `code_execution_config` is default to False.

    This agent executes function calls.
    """

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant that searches web and generates report.
"""

    @staticmethod
    def get_functions_config() -> Dict[str, Any]:
        """Get the functions part of the llm_config for the agent.

        Returns:
            The functions part of the llm_config for the agent.
        """

        functions = {
            "search_web": {
                "name": "search_web",
                "description": "search the web for the user and provide the search report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query to search",
                        }
                    },
                    "required": ["query"],
                },
            }
        }

        return functions

    @staticmethod
    def get_function_map(google_api_key: str, google_cse_id: str) -> Dict[str, Any]:
        """Get the function_map for the agent.

        Args:
            google_api_key: The api_key for the agent.
            google_cse_id: Google Custom Search Engine ID

        Returns:
            The function_map for the agent.
        """

        def search_web(
            query: str,
            *,
            api_key: str = google_api_key,
            google_cse_id: str = google_cse_id,
        ) -> List[Dict[str, Any]]:
            """Search the web for the user and provide the search report.

            Args:
                query: The query to search.

            Returns:
                The search report.
            """

            # Build a service object for the API
            service = build("customsearch", "v1", developerKey=api_key)

            # Perform the search
            res = service.cse().list(q=query, cx=google_cse_id).execute()

            # Return the results
            items: List[Dict[str, Any]] = res.get("items", [])
            return items

        function_map = {"search_web": search_web}
        return function_map

    @staticmethod
    def get_llm_config(
        config_list: List[Dict[str, Any]], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get the llm_config for the agent.

        Args:
            config_list: The config_list for the agent.
            timeout: The timeout for the agent.

        Returns:
            The llm_config for the agent.
        """

        llm_config = {
            "functions": GoogleSearchAgent.get_functions_config(),
            "config_list": config_list,
            "timeout": timeout,
        }

        return llm_config

    def __init__(
        self,
        name: str,
        *,
        google_api_key: str,
        google_cse_id: str,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        is_termination_msg: Optional[Callable[[Dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict[str, Any], bool]] = False,
        # llm_config: Optional[Union[Dict, bool]] = None,
        config_list: List[Dict[str, Any]],
        timeout: Optional[int] = None,
        default_auto_reply: Optional[Union[str, Dict[str, Any], None]] = "",
    ) -> None:
        """
        Initialize the GoogleSearchAgent.

        Args:
            name (str): The name of the agent.
            google_api_key (str): The API key to use for Google Search.
            google_cse_id (str): The Custom Search Engine ID to use for Google Search.
            system_message (Optional[str]): The default system message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            is_termination_msg (Optional[Callable[[Dict[str, Any]], bool]]): A function to determine if a message should terminate the agent. Defaults to None.
            max_consecutive_auto_reply (Optional[int]): The maximum number of consecutive auto replies. Defaults to None.
            human_input_mode (Optional[str]): The mode for human input. Defaults to "NEVER".
            code_execution_config (Optional[Union[Dict[str, Any], bool]]): The configuration for code execution. Defaults to False.
            config_list (List[Dict[str, Any]]): The list of configurations for the agent.
            timeout (Optional[int]): The timeout for the agent. Defaults to None.
            default_auto_reply (Optional[Union[str, Dict[str, Any], None]]): The default auto reply for the agent. Defaults to "".
        """
        llm_config = GoogleSearchAgent.get_llm_config(config_list, timeout)
        function_map = GoogleSearchAgent.get_function_map(google_api_key, google_cse_id)
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id

        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
