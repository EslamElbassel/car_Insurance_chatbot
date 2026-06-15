"""
app.py — AROPE Motor Insurance Chatbot (Chainlit UI)
=====================================================
Run with:
    chainlit run app.py

Install dependencies:
    pip install chainlit langchain langchain-openai langchain-community faiss-cpu pypdf
"""

import os
from pathlib import Path

import chainlit as cl
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
PDF_PATH        = "AMG_cond.pdf"
FAISS_INDEX_DIR = "motors_terms_index"
INDEX_NAME      = "motors_terms_index"
NUM_RESULTS     = 4

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """\
أنت "مساعد AROPE للتأمين على السيارات"، مساعد ذكاء اصطناعي متخصص تابع لشركة AROPE للتأمين في مصر.
دورك هو مساعدة العملاء في فهم شروط وأحكام وثيقة التأمين على السيارات بدقة وأمانة،
بالاعتماد الكامل على المعلومات المسترجعة من الوثيقة الرسمية فقط.

## قواعد صارمة
1. لا تُجب أبداً من معرفتك العامة — اعتمد حصرياً على المعلومات المسترجعة.
2. إذا لم تجد الإجابة قل: "لا تتوفر لديّ معلومات كافية. يُرجى التواصل مع فريق خدمة العملاء."
3. اذكر رقم البند أو القسم المرجعي إن وُجد.
4. أنهِ إجابتك بسؤال مفتوح لتشجيع العميل على الاستفسار أكثر.
5. تستقبل الأسئلة باللغة العربية أو الإنجليزية وتُجيب بنفس لغة العميل.
"""

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
{system_message}

## الوثائق المسترجعة
{context}

## سؤال العميل
{question}

## إجابتك
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY,
    )


def build_index(embeddings: OpenAIEmbeddings) -> None:
    loader = PyPDFLoader(PDF_PATH)
    docs   = loader.load()
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(folder_path=FAISS_INDEX_DIR, index_name=INDEX_NAME)


def load_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    index_file = Path(FAISS_INDEX_DIR) / f"{INDEX_NAME}.faiss"
    if not index_file.exists():
        build_index(embeddings)
    return FAISS.load_local(
        folder_path=FAISS_INDEX_DIR,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def build_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RESULTS})
    prompt    = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm       = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        streaming=True,           # enables token-by-token streaming in Chainlit
    )
    return (
        {
            "context":        retriever | format_docs,
            "question":       RunnablePassthrough(),
            "system_message": lambda _: SYSTEM_MESSAGE,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# ---------------------------------------------------------------------------
# Chainlit hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Runs once when a user opens the chat. Sets up the chain."""

    # Show a loading message while the index is prepared
    msg = cl.Message(content="")
    await msg.send()
    msg.content = "⏳ جارٍ تحميل وثيقة التأمين وإعداد المساعد..."
    await msg.update()

    embeddings  = build_embeddings()
    vectorstore = load_vectorstore(embeddings)
    chain       = build_chain(vectorstore)

    # Store the chain in the user session so it persists across messages
    cl.user_session.set("chain", chain)

    msg.content = (
        "مرحباً! أنا **مساعد AROPE للتأمين على السيارات** 🚗\n\n"
        "يمكنني مساعدتك في الاستفسار عن شروط وأحكام وثيقة التأمين على السيارات.\n\n"
        "كيف يمكنني مساعدتك؟"
    )
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Runs on every user message. Streams the response token by token."""

    chain = cl.user_session.get("chain")

    # Create an empty message that we'll stream tokens into
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Stream tokens from the chain
    async for token in chain.astream(message.content):
        await response_msg.stream_token(token)

    # Finalize the message
    await response_msg.update()
