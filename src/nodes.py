from colorama import Fore, Style
from .agents import Agents
from .tools.QQMailTools import QQMailTools
from .tools.GmailTools import GmailToolsClass
from .tools.NeteaseEmailTools import NeteaseEmailTools
from .state import GraphState, Email
from .context.context_manager import ContextManager
from .memory.sender_memory import SenderMemoryManager, DatabaseMemoryStore
import os


def get_email_tools():
    """根据配置获取邮件工具"""
    use_gmail = os.getenv('USE_GMAIL', 'false').lower() == 'true'
    if use_gmail:
        return GmailToolsClass()
    
    # 非 Gmail 模式，根据 EMAIL_PROVIDER 选择
    email_provider = os.getenv('EMAIL_PROVIDER', 'qq').lower()
    if email_provider == '163' or email_provider == 'netease':
        return NeteaseEmailTools()
    else:
        # 默认使用 QQ 邮箱
        return QQMailTools()


class Nodes:
    def __init__(self):
        self.agents = Agents()
        self.email_tools = get_email_tools()
        memory_store = DatabaseMemoryStore()
        self.sender_memory = SenderMemoryManager(store=memory_store)
        self.context_manager = ContextManager(sender_memory=self.sender_memory)

    def load_new_emails(self, state: GraphState) -> GraphState:
        """Loads new emails from QQ Mail and updates the state."""
        print(Fore.YELLOW + "Loading new emails...\n" + Style.RESET_ALL)
        recent_emails = self.email_tools.fetch_unanswered_emails()
        emails = [Email(**email) for email in recent_emails]
        return {"emails": emails}

    def check_new_emails(self, state: GraphState) -> str:
        """Checks if there are new emails to process."""
        if len(state['emails']) == 0:
            print(Fore.RED + "No new emails" + Style.RESET_ALL)
            return "empty"
        else:
            print(Fore.GREEN + "New emails to process" + Style.RESET_ALL)
            return "process"

    def is_email_inbox_empty(self, state: GraphState) -> GraphState:
        return state

    def categorize_email(self, state: GraphState) -> GraphState:
        """Categorizes the current email using the categorize_email agent."""
        print(Fore.YELLOW + "Checking email category...\n" + Style.RESET_ALL)

        # Get the last email
        current_email = state["emails"][-1]
        result = self.agents.categorize_email.invoke({"email": current_email.body})
        print(Fore.MAGENTA + f"Email category: {result.category.value}" + Style.RESET_ALL)

        return {
            "email_category": result.category.value,
            "current_email": current_email
        }

    def route_email_based_on_category(self, state: GraphState) -> str:
        """Routes the email based on its category."""
        print(Fore.YELLOW + "Routing email based on category...\n" + Style.RESET_ALL)
        category = state["email_category"]
        if category == "product_enquiry":
            return "product related"
        elif category == "unrelated":
            return "unrelated"
        else:
            return "not product related"

    def construct_rag_queries(self, state: GraphState) -> GraphState:
        """Constructs RAG queries based on the email content."""
        print(Fore.YELLOW + "Designing RAG query...\n" + Style.RESET_ALL)
        email_content = state["current_email"].body
        query_result = self.agents.design_rag_queries.invoke({"email": email_content})

        return {"rag_queries": query_result.queries}

    def retrieve_from_rag(self, state: GraphState) -> GraphState:
        """两步式 RAG：先用 src.Rag.retriever 拿 context，再调 LLM 答。"""
        print(Fore.YELLOW + "Retrieving information from internal knowledge...\n" + Style.RESET_ALL)
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        retriever = getattr(self.agents, "retriever", None)
        if retriever is None:
            print(Fore.RED + "RAG retriever not available; skipping retrieval." + Style.RESET_ALL)
            return {"retrieved_documents": ""}

        final_answer = ""
        for query in state["rag_queries"]:
            # 1) 显式从 src.Rag 拿 context 块
            docs = retriever.invoke(query)
            context_text = "\n\n".join(doc.page_content for doc in docs)

            # 2) 调 LLM 基于 context 答（结构与原 generate_rag_answer 链一致）
            chain = (
                {"context": lambda _: context_text, "question": RunnablePassthrough()}
                | self.agents.qa_prompt
                | self.agents.llm
                | StrOutputParser()
            )
            rag_result = chain.invoke(query)
            final_answer += query + "\n" + rag_result + "\n\n"

        return {"retrieved_documents": final_answer}

    def assemble_context(self, state: GraphState) -> GraphState:
        """Assembles all context layers into a single bundle."""
        print(Fore.YELLOW + "Assembling context bundle...\n" + Style.RESET_ALL)
        assembled = self.context_manager.build_context_bundle(state, self.sender_memory)
        return {"assembled_context": assembled}

    def write_draft_email(self, state: GraphState) -> GraphState:
        """Writes a draft email based on the current email and retrieved information."""
        print(Fore.YELLOW + "Writing draft email...\n" + Style.RESET_ALL)

        # Get assembled context and messages history
        assembled_context = state.get("assembled_context", "")
        writer_messages = state.get("writer_messages", [])

        # Write email
        draft_result = self.agents.email_writer.invoke({
            "context": assembled_context,
            "email_category": state["email_category"],
            "email_content": state["current_email"].body,
            "history": writer_messages
        })
        email = draft_result.email
        trials = state.get("trials", 0) + 1

        # Append writer's draft to the message list
        writer_messages.append(f"**Draft {trials}:**\n{email}")

        return {
            "generated_email": email,
            "trials": trials,
            "writer_messages": writer_messages
        }

    def verify_generated_email(self, state: GraphState) -> GraphState:
        """Verifies the generated email using the proofreader agent."""
        print(Fore.YELLOW + "Verifying generated email...\n" + Style.RESET_ALL)
        review = self.agents.email_proofreader.invoke({
            "initial_email": state["current_email"].body,
            "generated_email": state["generated_email"],
        })

        writer_messages = state.get('writer_messages', [])
        writer_messages.append(f"**Proofreader Feedback:**\n{review.feedback}")

        return {
            "sendable": review.send,
            "writer_messages": writer_messages
        }

    def must_rewrite(self, state: GraphState) -> str:
        """Determines if the email needs to be rewritten based on the review and trial count."""
        email_sendable = state["sendable"]
        if email_sendable:
            print(Fore.GREEN + "Email is good, ready to be sent!!!" + Style.RESET_ALL)
            state["emails"].pop()
            state["writer_messages"] = []
            return "send"
        elif state["trials"] >= 3:
            print(Fore.RED + "Email is not good, we reached max trials must stop!!!" + Style.RESET_ALL)
            state["emails"].pop()
            state["writer_messages"] = []
            return "stop"
        else:
            print(Fore.RED + "Email is not good, must rewrite it..." + Style.RESET_ALL)
            return "rewrite"

    def save_draft_email(self, state: GraphState) -> GraphState:
        """
        Saves the email reply as a draft (saved as .eml file in drafts directory).
        """
        print(Fore.YELLOW + "Saving email draft...\n" + Style.RESET_ALL)

        try:
            current_email = state["current_email"]
            reply_text = state["generated_email"]

            result = self.email_tools.create_draft_reply(current_email, reply_text)

            if result:
                if isinstance(result, dict) and result.get("draft_path"):
                    print(Fore.GREEN + f"Draft saved successfully to: {result['draft_path']}" + Style.RESET_ALL)
                elif isinstance(result, dict) and result.get("draft_id"):
                    print(Fore.GREEN + f"Draft saved successfully: {result['draft_id']}" + Style.RESET_ALL)
                else:
                    print(Fore.GREEN + "Draft saved successfully." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Failed to save draft." + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"Error saving draft: {e}" + Style.RESET_ALL)

        # Reset state for next email
        return {"assembled_context": "", "retrieved_documents": "", "trials": 0}

    def skip_unrelated_email(self, state):
        """Skip unrelated email and remove from emails list."""
        print("Skipping unrelated email...\n")
        state["emails"].pop()
        return state
