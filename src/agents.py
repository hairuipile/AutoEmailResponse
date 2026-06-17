from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import ZhipuAIEmbeddings
import os
from .structure_outputs import *
from .prompts import *

class Agents():
    def __init__(self):
        llm_provider = os.getenv("LLM_PROVIDER", "DEEPSEEK")
        
        if llm_provider == "DEEPSEEK":
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
            llm = ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.1, 
                api_key=deepseek_api_key
            )
            # Embedding 用智普
            zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY", "")
            embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=zhipuai_api_key)
        elif llm_provider == "ZHIPUAI":
            zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY", "")
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=zhipuai_api_key)
            llm = ZhipuAI(model="glm-4", temperature=0.1)
            embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=zhipuai_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        email_category_prompt = PromptTemplate(
            template=CATEGORIZE_EMAIL_PROMPT, 
            input_variables=["email"]
        )
        self.categorize_email = (
            email_category_prompt | 
            llm.with_structured_output(CategorizeEmailOutput)
        )

        generate_query_prompt = PromptTemplate(
            template=GENERATE_RAG_QUERIES_PROMPT, 
            input_variables=["email"]
        )
        self.design_rag_queries = (
            generate_query_prompt | 
            llm.with_structured_output(RAGQueriesOutput)
        )
        
        qa_prompt = ChatPromptTemplate.from_template(GENERATE_RAG_ANSWER_PROMPT)
        self.generate_rag_answer = (
            {"context": retriever, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        writer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EMAIL_WRITER_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "# **EMAIL CATEGORY:** {email_category}\n\n# **EMAIL CONTENT:**\n{email_content}")
            ]
        )
        self.email_writer = (
            writer_prompt |
            llm.with_structured_output(WriterOutput)
        )

        proofreader_prompt = PromptTemplate(
            template=EMAIL_PROOFREADER_PROMPT, 
            input_variables=["initial_email", "generated_email"]
        )
        self.email_proofreader = (
            proofreader_prompt | 
            llm.with_structured_output(ProofReaderOutput) 
        )
