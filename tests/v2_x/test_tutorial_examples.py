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

import os

from nemoguardrails import RailsConfig
from tests.utils import TestChat

ROOT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..")


def test_hello_world_1():
    config = RailsConfig.from_path(
        os.path.join(ROOT_FOLDER, "examples", "v2_x", "tutorial", "hello_world_1")
    )
    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello World!"


def test_hello_world_2():
    config = RailsConfig.from_path(
        os.path.join(ROOT_FOLDER, "examples", "v2_x", "tutorial", "hello_world_2")
    )
    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "Hello world!"


def test_hello_world_3():
    config = RailsConfig.from_path(
        os.path.join(ROOT_FOLDER, "examples", "v2_x", "tutorial", "hello_world_3")
    )
    chat = TestChat(
        config,
        llm_completions=[" user express greeting"],
    )

    chat >> "hi there!"
    chat << "Hello world!"


def test_guardrails_1():
    config = RailsConfig.from_path(
        os.path.join(ROOT_FOLDER, "examples", "v2_x", "tutorial", "guardrails_1")
    )
    chat = TestChat(
        config,
        llm_completions=["$is_safe = True", "$is_safe = False"],
    )

    chat >> "hi"
    chat << "Hello world!"

    chat >> "you are stupid"
    chat << "I'm sorry, I can't respond to that."
