# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat

PATRONUS_EVALUATE_API_URL = "https://api.patronus.ai/v1/evaluate"
COLANG_CONFIG = """
define user express greeting
  "hi"
define bot refuse to respond
  "I'm sorry, I can't respond to that."
"""

YAML_PREFIX = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
rails:
  output:
    flows:
      - patronus api check output
"""


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
def test_patronus_evaluate_api_success_strategy_all_pass(monkeypatch):
    """
    Test that the "all_pass" success strategy passes when all evaluators pass
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "all_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    print("rails config", YAML_PREFIX + yaml_evaluate_config)
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_evaluate_api_success_strategy_all_pass_fails_when_one_failure(
    monkeypatch,
):
    """
    Test that the "all_pass" success strategy fails when only one evaluator fails
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "all_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


def test_patronus_evaluate_api_success_strategy_any_pass_passes_when_one_failure(
    monkeypatch,
):
    """
    Test that the "any_pass" success strategy passes when only one evaluator fails
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "Hi there! How are you doing?"


def test_patronus_evaluate_api_success_strategy_any_pass_fails_when_all_fail(
    monkeypatch,
):
    """
    Test that the "any_pass" success strategy fails when all evaluators fail
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


def test_patronus_evaluate_api_internal_error_when_no_env_set():
    """
    Test that an internal error is returned when the PATRONUS_API_KEY variable is not set
    """
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


def test_patronus_evaluate_api_internal_error_when_no_evaluators_provided():
    """
    Test that an internal error is returned when no 'evaluators' dict
    is passed in teh evaluate_config params.
    """
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


def test_patronus_evaluate_api_internal_error_when_evaluator_dict_does_not_have_evaluator_key():
    """
    Test that an internal error is returned when the passed evaluator dict in the
    evaluator_config does not have the 'evaluator' key.
    """
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


@pytest.mark.asyncio
def test_patronus_evaluate_api_default_success_strategy_is_all_pass_happy_case(
    monkeypatch,
):
    """
    Test that when the success strategy is omitted, the default "all_pass" is chosen,
    and thus the request passes since all evaluators pass.
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    print("rails config", YAML_PREFIX + yaml_evaluate_config)
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_evaluate_api_default_success_strategy_all_pass_fails_when_one_failure(
    monkeypatch,
):
    """
    Test that when the success strategy is omitted, the default "all_pass" is chosen,
    and thus the request fails since one evaluator also fails.
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    print("rails config", YAML_PREFIX + yaml_evaluate_config)
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


@pytest.mark.asyncio
def test_patronus_evaluate_api_internal_error_when_400_status_code(
    monkeypatch,
):
    """
    Test that when the API returns a 4XX status code,
    the bot returns an internal error response
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    print("rails config", YAML_PREFIX + yaml_evaluate_config)
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            status=400,
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


@pytest.mark.asyncio
def test_patronus_evaluate_api_default_response_when_500_status_code(
    monkeypatch,
):
    """
    Test that when the API returns a 5XX status code,
    the bot returns the default fail response
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            status=500,
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."
