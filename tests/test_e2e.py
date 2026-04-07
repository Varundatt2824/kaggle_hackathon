"""
End-to-end test script for MedSimplify.

Run with: uv run python -m tests.test_e2e

This script:
1. Creates a sample lab report (text)
2. Runs it through the full analysis graph
3. Prints the explanation and follow-up questions
4. Optionally tests the chat follow-up

Prerequisites: Ollama running locally with Gemma 4 model.
If you don't have gemma4:31b, you can test with a smaller model
by setting OLLAMA_MODEL=gemma3:8b in your .env file.
"""

from langchain_core.messages import HumanMessage

from agent.graph import build_analysis_graph, build_chat_graph


# --- Sample lab report for testing ---
SAMPLE_LAB_REPORT = """
PATIENT: Jane Doe
DATE: 2026-04-01
ORDERING PHYSICIAN: Dr. Smith

COMPLETE BLOOD COUNT (CBC)
Test                Value       Units       Reference Range     Flag
White Blood Cells   11.2        10^3/uL     4.5-11.0           HIGH
Red Blood Cells     4.85        10^6/uL     4.00-5.50
Hemoglobin          14.2        g/dL        12.0-17.5
Hematocrit          42.1        %           36.0-52.0
Platelets           245         10^3/uL     150-400

LIPID PANEL
Test                Value       Units       Reference Range     Flag
Total Cholesterol   242         mg/dL       <200                HIGH
LDL Cholesterol     165         mg/dL       <100                HIGH
HDL Cholesterol     38          mg/dL       >40                 LOW
Triglycerides       195         mg/dL       <150                HIGH

METABOLIC PANEL
Test                Value       Units       Reference Range     Flag
Glucose (Fasting)   108         mg/dL       70-100              HIGH
BUN                 18          mg/dL       7-20
Creatinine          1.0         mg/dL       0.7-1.3
"""


def main():
    print("=" * 60)
    print("MedSimplify — End-to-End Test")
    print("=" * 60)

    # Build the analysis graph
    print("\n[1/3] Building analysis graph...")
    analysis_graph = build_analysis_graph()

    # Run the analysis
    print("[2/3] Analyzing sample lab report...")
    print("      (This may take a minute on first run)\n")

    initial_state = {
        "raw_input": SAMPLE_LAB_REPORT,
        "input_type": "text",
        "file_name": "sample_lab_report.txt",
        "extracted_text": "",
        "report_type": "",
        "detected_language": "English",
        "output_language": "English",
        "findings": [],
        "explanation": "",
        "followup_questions": [],
        "chat_history": [],
    }

    result = analysis_graph.invoke(initial_state)

    # Display results
    print("=" * 60)
    print("REPORT TYPE:", result["report_type"])
    print("=" * 60)

    print("\n--- FINDINGS ---")
    for i, finding in enumerate(result["findings"], 1):
        print(f"\n  [{i}] {finding}")

    print("\n\n--- EXPLANATION ---")
    print(result["explanation"])

    print("\n\n--- SUGGESTED QUESTIONS FOR YOUR DOCTOR ---")
    for i, q in enumerate(result["followup_questions"], 1):
        print(f"  {i}. {q}")

    # Test chat follow-up
    print("\n" + "=" * 60)
    print("[3/3] Testing chat follow-up...")
    print("=" * 60)

    chat_graph = build_chat_graph()

    # Simulate a user question
    test_question = "What does my high LDL cholesterol mean? Should I be worried?"
    print(f"\nUser: {test_question}")

    chat_state = {
        **result,
        "chat_history": [HumanMessage(content=test_question)],
    }

    chat_result = chat_graph.invoke(chat_state)

    # The last message in chat_history is the AI's response
    ai_response = chat_result["chat_history"][-1]
    print(f"\nMedSimplify: {ai_response.content}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
