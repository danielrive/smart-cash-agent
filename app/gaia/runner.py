import os
import gradio as gr
import requests
import pandas as pd
import re

from app.agent.agent import run_agent 

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def clean_answer(text: str) -> str:
    """
    Remove <thinking> or similar tags from the model output.
    Keep only the final user-facing answer.
    """
    # remove <thinking>...</thinking> sections
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # strip whitespace
    return text.strip()

# --- Wrap your LangGraph agent in a callable class ---
class GAIAAgent:
    def __init__(self):
        print("GAIAAgent initialized.")

    def __call__(self, question: str) -> str:
        """
        GAIA will call this function with a question (string).
        We pass it to our LangGraph agent and return the answer.
        """
        print(f"GAIAAgent received question: {question[:50]}...")
        try:
            response = run_agent(question)  # 👈 runs your graph with a message
            raw_answer = response["messages"][-1].content  # extract final text
            answer = clean_answer(raw_answer)
            
        except Exception as e:
            answer = f"Error: {e}"
        print(f"Returning: {answer[:50]}...")
        return answer


def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        return "Please Login to Hugging Face.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GAIAAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log, answers_payload = [], []

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        submitted_answer = agent(question_text)
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"✅ Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
        )
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation")
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
