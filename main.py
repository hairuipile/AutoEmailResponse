from colorama import Fore, Style
from dotenv import load_dotenv
from pathlib import Path
import os

# Load all env variables FIRST - explicitly specify .env file
load_dotenv(Path(__file__).parent / ".env")

# Now import and initialize workflow
from src.graph import Workflow
from src.Rag.preprocess import check_and_update

# config
config = {'recursion_limit': 100}

# 检查知识库变化，如有变化则更新向量库
print(Fore.CYAN + "Checking knowledge base for updates..." + Style.RESET_ALL)
check_and_update()

workflow = Workflow()
app = workflow.app

initial_state = {
    "emails": [],
    "current_email": {
      "id": "",
      "threadId": "",
      "messageId": "",
      "references": "",
      "sender": "",
      "subject": "",
      "body": ""
    },
    "sender_key": "",
    "sender_strategy": "",
    "email_category": "",
    "generated_email": "",
    "rag_queries": [],
    "retrieved_documents": "",
    "top_level_rules": "",
    "long_term_memory": "",
    "context_summary": "",
    "selected_context": "",
    "assembled_context": "",
    "context_token_budget": 0,
    "writer_messages": [],
    "sendable": False,
    "trials": 0
}

# Run the automation
print(Fore.GREEN + "Starting workflow..." + Style.RESET_ALL)
for output in app.stream(initial_state, config):
    for key, value in output.items():
        print(Fore.CYAN + f"Finished running: {key}:" + Style.RESET_ALL)


