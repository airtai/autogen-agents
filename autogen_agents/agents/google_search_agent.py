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
    def get_function_map(api_key: str, cse_id: str) -> Dict[str, Any]:
        """Get the function_map for the agent.

        Args:
            api_key: The api_key for the agent.
            cse_id: Google Custom Search Engine ID

        Returns:
            The function_map for the agent.
        """

        def search_web(
            query: str, *, api_key: str = api_key, cse_id: str = cse_id
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
            res = service.cse().list(q=query, cx=cse_id).execute()

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
        api_key: str,
        cse_id: str,
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
        llm_config = GoogleSearchAgent.get_llm_config(config_list, timeout)
        function_map = GoogleSearchAgent.get_function_map(api_key, cse_id)
        self.api_key = api_key

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
