from __future__ import annotations

import json
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_zenith_inkling_agent import SampledTurn, run_agent_loop
from tinker_cookbook.renderers import Message, ToolCall, UnparsedToolCall


SUCCESS_RESULT = json.dumps({
    "exit_code": 0,
    "stdout": "corrected\n",
    "error": None,
    "evidence_exit_code": 0,
}, separators=(",", ":"))
PROGRAM_ERROR_RESULT = json.dumps({
    "exit_code": 1,
    "stdout": "NameError: name 'invented_helper' is not defined\n",
    "error": "NameError: name 'invented_helper' is not defined\n",
    "evidence_exit_code": 0,
}, separators=(",", ":"))


def tool_message(name: str, arguments: str, call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(
            id=call_id,
            function=ToolCall.FunctionBody(name=name, arguments=arguments),
        )],
    )


def sampled(message: Message, turn: int, termination: str = "stop_sequence") -> SampledTurn:
    return SampledTurn(message, 4, termination, {"turn": turn})


class MultiTurnRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def run_sequence(
        self,
        bad_messages: list[tuple[Message, str]],
        tool_results: list[str] | None = None,
    ) -> tuple[dict, list[list[Message]], list[dict]]:
        observations: list[list[Message]] = []
        executions: list[dict] = []
        turns = [message for message, _ in bad_messages] + [
            tool_message(
                "browser_harness",
                json.dumps({"code": "print('corrected')"}),
                "corrected-call",
            ),
            Message(role="assistant", content="successful final answer"),
        ]
        results = iter(tool_results or [SUCCESS_RESULT])

        async def fake_sample(messages: list[Message], remaining: int, turn: int) -> SampledTurn:
            observations.append(list(messages))
            return sampled(turns[turn - 1], turn)

        async def fake_execute(code: str, turn: int, call: int) -> str:
            executions.append({"code": code, "turn": turn, "call": call})
            return next(results)

        result = await run_agent_loop(
            [Message(role="user", content="task")],
            fake_sample,
            fake_execute,
            32000,
        )
        self.assertEqual(result["termination_reason"], "final_answer")
        self.assertEqual(result["final_response"], "successful final answer")
        self.assertEqual(result["model_turns"], len(turns))
        self.assertEqual(result["total_generated_tokens"], 4 * len(turns))
        self.assertEqual(result["attempted_tool_calls"], len(bad_messages) + 1)
        self.assertEqual(result["error_feedback_turns"], len(bad_messages))
        self.assertEqual(
            [event["turn"] for event in result["events"] if event["type"] == "assistant"],
            list(range(1, len(turns) + 1)),
        )
        self.assertEqual(
            sum(
                event["type"] == "error_feedback"
                or (event["type"] == "tool_result" and event["is_error"])
                for event in result["events"]
            ),
            len(bad_messages),
        )
        for index, (_, exact_error) in enumerate(bad_messages, start=1):
            feedback = observations[index][-1]
            self.assertEqual(feedback["content"], exact_error)
        return result, observations, executions

    async def test_invalid_python_is_returned_then_corrected(self) -> None:
        exact_error = (
            "invalid browser_harness arguments: SyntaxError: invalid syntax "
            "(<inkling-browser-harness>, line 1)"
        )
        result, _, executions = await self.run_sequence([(
            tool_message(
                "browser_harness",
                json.dumps({"code": 'c dp("Accessibility.getFullAXTree")'}),
                "syntax-call",
            ),
            exact_error,
        )])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(executions, [{
            "code": "print('corrected')\n",
            "turn": 2,
            "call": 1,
        }])

    async def test_invalid_argument_json_and_schema_are_returned_then_corrected(self) -> None:
        json_error = (
            "invalid browser_harness arguments: JSONDecodeError: Expecting property name "
            "enclosed in double quotes: line 1 column 2 (char 1)"
        )
        schema_error = (
            "invalid browser_harness arguments: ValueError: arguments must be an object "
            "containing only a string 'code' field"
        )
        result, _, executions = await self.run_sequence([
            (tool_message("browser_harness", "{", "json-call"), json_error),
            (tool_message(
                "browser_harness",
                json.dumps({"code": 7, "extra": True}),
                "schema-call",
            ), schema_error),
        ])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(len(executions), 1)

    async def test_unknown_tool_is_returned_then_corrected(self) -> None:
        result, _, executions = await self.run_sequence([(
            tool_message("invented_helper", "{}", "unknown-call"),
            "unknown tool 'invented_helper'",
        )])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(len(executions), 1)

    async def test_unparsed_tool_xml_error_is_returned_then_corrected(self) -> None:
        exact_error = "mismatched closing tag for parameter code"
        malformed = Message(
            role="assistant",
            content="recoverable malformed tool content",
            unparsed_tool_calls=[UnparsedToolCall(
                raw_text="<tool_call><function=browser_harness>broken</tool_call>",
                error=exact_error,
            )],
        )
        result, observations, executions = await self.run_sequence(
            [(malformed, exact_error)]
        )
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(len(executions), 1)
        self.assertEqual(
            observations[1][-2]["content"], "recoverable malformed tool content"
        )

    async def test_program_execution_error_is_returned_then_corrected(self) -> None:
        bad_call = tool_message(
            "browser_harness",
            json.dumps({"code": "invented_helper()"}),
            "program-error-call",
        )
        result, _, executions = await self.run_sequence(
            [(bad_call, PROGRAM_ERROR_RESULT)],
            [PROGRAM_ERROR_RESULT, SUCCESS_RESULT],
        )
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(len(executions), 2)
        self.assertEqual(executions[0]["code"], "invented_helper()\n")

    async def test_protocol_or_sandbox_loss_remains_terminal(self) -> None:
        observations = []

        async def fake_sample(messages: list[Message], remaining: int, turn: int) -> SampledTurn:
            observations.append(list(messages))
            return sampled(tool_message(
                "browser_harness",
                json.dumps({"code": "print(page_info())"}),
                "lost-sandbox-call",
            ), turn)

        async def lost_sandbox(code: str, turn: int, call: int) -> str:
            raise ConnectionError("sandbox protocol stream closed")

        result = await run_agent_loop(
            [Message(role="user", content="task")],
            fake_sample,
            lost_sandbox,
            32000,
        )
        self.assertEqual(result["termination_reason"], "irrecoverable_infrastructure_error")
        self.assertEqual(
            result["error"],
            "browser_harness protocol failed: ConnectionError: sandbox protocol stream closed",
        )
        self.assertEqual(result["attempted_tool_calls"], 1)
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["error_feedback_turns"], 0)
        self.assertEqual(len(observations), 1)

    async def test_recoverable_errors_stop_at_exact_generation_cap(self) -> None:
        async def malformed(messages: list[Message], remaining: int, turn: int) -> SampledTurn:
            return SampledTurn(
                Message(
                    role="assistant",
                    content="",
                    unparsed_tool_calls=[UnparsedToolCall(
                        raw_text="broken",
                        error="recoverable malformed call",
                    )],
                ),
                16000,
                "malformed",
                {"turn": turn},
            )

        async def must_not_execute(code: str, turn: int, call: int) -> str:
            self.fail("unparsed calls must not execute")

        result = await run_agent_loop(
            [Message(role="user", content="task")],
            malformed,
            must_not_execute,
            32000,
        )
        self.assertEqual(result["termination_reason"], "generation_budget_32000")
        self.assertEqual(result["total_generated_tokens"], 32000)
        self.assertEqual(result["model_turns"], 2)
        self.assertEqual(result["attempted_tool_calls"], 1)
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["error_feedback_turns"], 1)

    async def test_rollout_timeout_is_a_preserved_model_outcome(self) -> None:
        async def slow_sample(
            messages: list[Message], remaining: int, turn: int
        ) -> SampledTurn:
            await asyncio.sleep(0.05)
            return sampled(Message(role="assistant", content="late"), turn)

        async def must_not_execute(code: str, turn: int, call: int) -> str:
            self.fail("timed-out sampling must not execute a tool")

        result = await run_agent_loop(
            [Message(role="user", content="task")],
            slow_sample,
            must_not_execute,
            32000,
            timeout_seconds=0.001,
        )
        self.assertEqual(result["termination_reason"], "rollout_timeout_0.001")
        self.assertIsNone(result["error"])
        self.assertEqual(result["model_turns"], 1)
        self.assertEqual(result["total_generated_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
