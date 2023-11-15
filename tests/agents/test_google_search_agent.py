import os
from unittest import mock

import pytest
from autogen import config_list_from_json

from autogen_agents.agents._google_search_agent import GoogleSearchAgent


class TestGoogleSearchAgent:
    def test_agents_level_import(self) -> None:
        from autogen_agents.agents import (
            GoogleSearchAgent as AgentsLevelGoogleSearchAgent,
        )

        assert AgentsLevelGoogleSearchAgent == GoogleSearchAgent

    def test_get_functions_config(self) -> None:
        actual = GoogleSearchAgent.get_functions_config()
        expected = {
            "search_web": {
                "name": "search_web",
                "description": "search the web for the user and provide the search report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "query to search"}
                    },
                    "required": ["query"],
                },
            }
        }
        print(GoogleSearchAgent.get_functions_config())

        assert actual == expected, actual

    def test_get_function_map(self) -> None:
        actual = GoogleSearchAgent.get_function_map("google_api_key", "google_cse_id")

        search_web_function = actual["search_web"]
        assert callable(search_web_function)

    @pytest.mark.vcr(filter_query_parameters=["key", "cx"])  # type: ignore
    def test_search(self) -> None:
        actual = GoogleSearchAgent.get_function_map(
            google_api_key="google_api_key",  # pragma: allowlist secret
            google_cse_id="google_cse_id",
        )

        search_web_function = actual["search_web"]

        result = search_web_function("unofficial autogen agents")

        assert len(result) > 0

    @mock.patch.dict(
        os.environ, {"OPENAI_API_KEY": "api_key"}  # pragma: allowlist secret
    )
    def test_get_llm_config(self) -> None:
        config_list = config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={
                "model": ["gpt-4"],
            },
        )
        llm_config = GoogleSearchAgent.get_llm_config(config_list, timeout=120)
        print(llm_config)
        expected = {
            "functions": {
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
            },
            "config_list": [],
            "timeout": 120,
        }
        assert llm_config == expected

    @mock.patch.dict(
        os.environ, {"OPENAI_API_KEY": "open_api_key"}  # pragma: allowlist secret
    )
    def test_init(self) -> None:
        config_list = config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={
                "model": ["gpt-4"],
            },
        )
        search_agent = GoogleSearchAgent(
            "search_agent",
            google_api_key="google_api_key",  # pragma: allowlist secret
            google_cse_id="google_cse_id",
            config_list=config_list,
            timeout=120,
        )
        assert (
            hasattr(search_agent, "google_api_key")
            and search_agent.google_api_key
            == "google_api_key"  # pragma: allowlist secret
        )
