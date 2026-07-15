import pytest
import torch

from scripts.validation.validate_qwen3_moe_manual_qa import (
    chat_template_token_ids,
    evaluate_answer,
)


@pytest.mark.parametrize(
    ("case_id", "answer"),
    (
        ("arithmetic", "391"),
        ("arithmetic", "The result is 391."),
        ("factual", "The capital of France is Paris."),
        ("format_following", '{"status":"ok","count":3}'),
        ("chinese_explanation", "MoE 通过路由为每个输入选择少量专家并组合专家输出。"),
    ),
)
def test_manual_qa_checks_accept_expected_answers(case_id, answer):
    passed, _criterion = evaluate_answer(case_id, answer)

    assert passed


@pytest.mark.parametrize(
    ("case_id", "answer"),
    (
        ("arithmetic", "392"),
        ("factual", "The capital is Lyon."),
        ("format_following", '```json\n{"status":"ok","count":3}\n```'),
        ("chinese_explanation", "这是一个模型。"),
    ),
)
def test_manual_qa_checks_reject_incorrect_answers(case_id, answer):
    passed, _criterion = evaluate_answer(case_id, answer)

    assert not passed


def test_manual_qa_check_rejects_unknown_case():
    with pytest.raises(ValueError, match="Unknown QA case_id"):
        evaluate_answer("unknown", "answer")


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ([1, 2, 3], [1, 2, 3]),
        ({"input_ids": [4, 5]}, [4, 5]),
        (torch.tensor([6, 7]), [6, 7]),
    ),
)
def test_chat_template_token_ids_accepts_supported_outputs(value, expected):
    assert chat_template_token_ids(value) == expected


def test_chat_template_token_ids_rejects_missing_input_ids():
    with pytest.raises(ValueError, match="without input_ids"):
        chat_template_token_ids({"attention_mask": [1]})


@pytest.mark.parametrize("value", ([], [1, "2"], "1 2"))
def test_chat_template_token_ids_rejects_invalid_sequences(value):
    with pytest.raises(TypeError, match="non-empty integer sequence"):
        chat_template_token_ids(value)
