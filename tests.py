import pytest
from unittest.mock import Mock
from utils import (
    generate_chosen_response,
    generate_modified_instruction_and_rejected_response,
    extract_modified_instruction,
    extract_rejected_response,
    generate_response_pair_and_modified_instruction,
    generate_multiple_judgements,
    generate_judgement,
    rejection_sample_judgements,
    generate_preference_data,
)
from datasets import Dataset

@pytest.fixture
def mock_model():
    model = Mock()
    model.generate.return_value = "Mock response"
    return model

def test_generate_chosen_response_success(mock_model):
    # Set up the mock to return a valid response
    mock_model.generate.return_value = "<answer>This is the chosen response</answer>"
    
    result = generate_chosen_response(mock_model, "Test instruction")
    
    # Check that the function returns the correct extracted response
    assert result == "This is the chosen response"
    
    # Verify that the model's generate method was called with the correct prompt
    mock_model.generate.assert_called_once()
    call_args = mock_model.generate.call_args[0][0]
    assert "Test instruction" in call_args

def test_generate_chosen_response_no_tags(mock_model):
    # Set up the mock to return a response without answer tags
    mock_model.generate.return_value = "This is not a properly formatted response"
    
    # Check that the function raises a ValueError when no answer tags are found
    with pytest.raises(ValueError, match="Chosen response not found in the raw response."):
        generate_chosen_response(mock_model, "Test instruction")

def test_generate_chosen_response_empty_answer(mock_model):
    # Set up the mock to return an empty answer
    mock_model.generate.return_value = "<answer></answer>"
    
    # Check that the function returns an empty string when the answer is empty
    result = generate_chosen_response(mock_model, "Test instruction")
    assert result == ""

def test_generate_chosen_response_multiple_answers(mock_model):
    # Set up the mock to return multiple answers
    mock_model.generate.return_value = "<answer>First answer</answer> <answer>Second answer</answer>"
    
    # Check that the function returns only the first answer
    result = generate_chosen_response(mock_model, "Test instruction")
    assert result == "First answer"

def test_generate_modified_instruction_and_rejected_response(mock_model):
    mock_model.generate.return_value = """
    User Question Modified
    Modified instruction

    The start of Assistant's answer to the modified instruction
    Rejected response
    The end of Assistant's answer to the modified instruction
    """
    modified_instruction, rejected_response = generate_modified_instruction_and_rejected_response(
        mock_model, "Original prompt", "Chosen response"
    )
    assert modified_instruction == "Modified instruction"
    assert rejected_response == "Rejected response"

def test_extract_modified_instruction():
    raw_response = """
    User Question Modified
    This is the modified instruction
    The start of Assistant's answer
    """
    result = extract_modified_instruction(raw_response)
    assert result == "This is the modified instruction"

def test_extract_rejected_response():
    raw_response = """
    The start of Assistant's answer to the modified instruction
    This is the rejected response
    The end of Assistant's answer to the modified instruction
    """
    result = extract_rejected_response(raw_response)
    assert result == "This is the rejected response"

def test_generate_response_pair_and_modified_instruction(mock_model):
    mock_model.generate.side_effect = [
        "Chosen response",
        """
        User Question Modified
        Modified instruction

        The start of Assistant's answer to the modified instruction
        Rejected response
        The end of Assistant's answer to the modified instruction
        """
    ]
    chosen, rejected, modified = generate_response_pair_and_modified_instruction(mock_model, "User instruction")
    assert chosen == "Chosen response"
    assert rejected == "Rejected response"
    assert modified == "Modified instruction"

def test_generate_multiple_judgements(mock_model):
    mock_model.generate.return_value = "[[A]]"
    results = generate_multiple_judgements(mock_model, "Test prompt", 3)
    assert results == ["A", "A", "A"]

@pytest.mark.parametrize("raw_response, expected", [
    ("[[A]]", "A"),
    ("[[B]]", "B"),
    ("Assistant A is better", "A"),
    ("Assistant B is better", "B"),
])
def test_generate_judgement(mock_model, raw_response, expected):
    mock_model.generate.return_value = raw_response
    result = generate_judgement(mock_model, "Test prompt")
    assert result == expected

def test_generate_judgement_invalid(mock_model):
    mock_model.generate.return_value = "Invalid response"
    with pytest.raises(ValueError):
        generate_judgement(mock_model, "Test prompt")

def test_rejection_sample_judgements():
    generated_judgements = ["A", "B", "A", "A", "B"]
    ground_truth = "A"
    result = rejection_sample_judgements(generated_judgements, ground_truth)
    assert result == ["A", "A", "A"]


if __name__ == "__main__":
    pytest.main()
