"""
AROPE Motors Insurance Chatbot
================================
Source document: AMG_cond.pdf (Aman Gold Motors Insurance Policy)
Flow components mapped:
  File (Read File)        →  PyPDFLoader
  SplitText               →  CharacterTextSplitter  (chunk_size=1000, overlap=200, sep="\n")
  EmbeddingModel          →  OpenAIEmbeddings       (text-embedding-3-large)
  FAISS                   →  FAISS vector store      (index: motors_terms_index)
  Parser                  →  format_docs helper
  Prompt                  →  ChatPromptTemplate
  LanguageModelComponent  →  ChatOpenAI             (gpt-5.4, temperature=0.1)
  ChatInput / ChatOutput  →  terminal chat loop
"""

import os
from pathlib import Path

from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import StdOutCallbackHandler
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PDF_PATH        = "AMG_cond.pdf"          # path to the motors policy PDF
FAISS_INDEX_DIR = "motors_terms_index"    # folder where the FAISS index is saved
INDEX_NAME      = "motors_terms_index"    # .faiss / .pkl file name inside the folder
NUM_RESULTS     = 4

# ---------------------------------------------------------------------------
# System message  (LanguageModelComponent → system_message field)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """\
أنت "مساعد AROPE للتأمين على السيارات"، مساعد ذكاء اصطناعي متخصص تابع لشركة AROPE للتأمين في مصر.
دورك هو مساعدة العملاء في فهم شروط وأحكام وثيقة التأمين على السيارات بدقة وأمانة،
بالاعتماد الكامل على المعلومات المسترجعة من الوثيقة الرسمية فقط.

---

## الهوية والنبرة

- اسمك: مساعد AROPE للتأمين على السيارات.
- تتحدث بأسلوب مهني، محايد، وودود يليق بموظف خدمة عملاء متخصص في التأمين.
- تستقبل الأسئلة باللغة العربية أو الإنجليزية، وتُجيب دائماً بنفس لغة العميل.
- لا تتبنى شخصية أخرى ولا تخرج عن دورك المحدد تحت أي ظرف.

---

## نطاق عملك

أنت مختص فقط بالإجابة على الأسئلة المتعلقة بـ:
- تغطيات وثيقة التأمين على السيارات وحدودها.
- الاستثناءات والحالات غير المشمولة بالوثيقة.
- إجراءات المطالبات والإبلاغ عن الحوادث.
- تعريفات المصطلحات التأمينية الواردة في الوثيقة.
- الشروط العامة والخاصة للوثيقة.

---

## قواعد صارمة يجب الالتزام بها

1. لا تُجب أبداً من معرفتك العامة — اعتمد حصرياً على المعلومات المسترجعة من الوثيقة.
2. إذا لم تجد الإجابة في الوثائق المسترجعة، قل بوضوح:
   "لا تتوفر لديّ معلومات كافية للإجابة على هذا السؤال في الوثيقة الحالية.
    يُرجى التواصل مع فريق خدمة العملاء للحصول على مساعدة متخصصة."
3. لا تُقدِّم وعوداً أو تأكيدات بشأن تسوية المطالبات — هذا من اختصاص فريق المطالبات.
4. لا تذكر أرقاماً أو نسباً أو مبالغ إلا إذا وردت صراحةً في الوثيقة المسترجعة.
5. إذا كان السؤال خارج نطاق التأمين على السيارات، اعتذر بلطف وأعِد توجيه العميل:
   "أنا متخصص فقط في التأمين على السيارات. للاستفسار عن منتجات أخرى،
    يُرجى التواصل مع فريقنا المختص."

---

## أسلوب الإجابة

- ابدأ إجابتك مباشرةً دون مقدمات زائدة.
- استخدم نقاطاً أو قوائم عند شرح بنود متعددة.
- اذكر رقم البند أو القسم المرجعي إن وُجد، مثل: "وفقاً للبند (3) من الشروط العامة...".
- إذا كانت الإجابة تحتمل استثناءات، نبّه إليها صراحةً.
- أنهِ إجابتك بسؤال مفتوح لتشجيع العميل على الاستفسار أكثر، مثل:
  "هل تودّ الاستفسار عن تفصيل آخر؟"

---

## حالات التصعيد

إذا أبدى العميل أياً مما يلي، أنهِ المحادثة بلطف وأحِله فوراً للدعم البشري:
- غضب أو استياء واضح.
- طلب التحدث مع موظف بشري.
- سؤال عن مطالبة قائمة بالفعل أو حادث محدد.
- معلومات شخصية أو بيانات وثيقة فردية.

رسالة التصعيد:
"يسعدني توصيلك بأحد متخصصينا الذين سيتمكنون من مساعدتك بشكل أفضل.
 يُرجى التواصل معنا على [رقم خدمة العملاء] أو عبر البريد الإلكتروني [البريد الإلكتروني]."
"""

# ---------------------------------------------------------------------------
# RAG prompt template  (Prompt component → template field)
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
{system_message}

---

## الوثائق المسترجعة

{context}

---

## سؤال العميل

{question}

---

## إجابتك
"""

# ---------------------------------------------------------------------------
# Step 1 – Embeddings model  (EmbeddingModel → text-embedding-3-large / OpenAI)
# ---------------------------------------------------------------------------

def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY,
    )

# ---------------------------------------------------------------------------
# Step 2 – Ingest pipeline: Load → Split → Embed → Save FAISS index
#           (File + SplitText + FAISS.from_documents)
# ---------------------------------------------------------------------------

def build_index(pdf_path: str, embeddings: OpenAIEmbeddings) -> None:
    """Load the PDF, split it, embed and save the FAISS index locally."""
    print(f"[Ingest] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Documents",documents)

    # SplitText settings from the flow
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        keep_separator=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"chunks {chunks}")
    print(f"[Ingest] Created {len(chunks)} chunks.")

    print("[Ingest] Building FAISS index …")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path=FAISS_INDEX_DIR, index_name=INDEX_NAME)
    print(f"[Ingest] Index saved to '{FAISS_INDEX_DIR}/{INDEX_NAME}'.")

# ---------------------------------------------------------------------------
# Step 3 – Load FAISS index
#           (FAISS.load_local with allow_dangerous_deserialization=True)
# ---------------------------------------------------------------------------

def load_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    index_file = Path(FAISS_INDEX_DIR) / f"{INDEX_NAME}.faiss"

    if not index_file.exists():
        print("[Setup] FAISS index not found — building from PDF …")
        build_index(PDF_PATH, embeddings)

    print("[Setup] Loading FAISS index …")
    return FAISS.load_local(
        folder_path=FAISS_INDEX_DIR,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,   # safe: index was built by us
    )

# ---------------------------------------------------------------------------
# Step 4 – Format retrieved docs  (Parser component → Stringify mode)
# ---------------------------------------------------------------------------

def format_docs(docs) -> str:
    """Join retrieved document chunks with a newline separator."""
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------------------------------
# Step 5 – Build the RAG chain
#           (Prompt → LanguageModelComponent → ChatOutput)
# ---------------------------------------------------------------------------

def build_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RESULTS})

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    llm = ChatOpenAI(
        model="gpt-4o",          # use gpt-4o as the production-available equivalent
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY,
        verbose=True
    )

    chain = (
        {
            "context":        retriever | format_docs,
            "question":       RunnablePassthrough(),
            "system_message": lambda _: SYSTEM_MESSAGE,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ---------------------------------------------------------------------------
# Step 6 – Chat loop  (ChatInput + ChatOutput)
# ---------------------------------------------------------------------------

def chat_loop(chain) -> None:
    print("\n" + "=" * 60)
    print("  مساعد AROPE للتأمين على السيارات")
    print("  AROPE Motor Insurance Assistant")
    print("=" * 60)
    print("اكتب سؤالك أو اكتب 'exit' للخروج.\n")

    callbacks = [StdOutCallbackHandler()] if True else []

    while True:
        user_input = input("أنت: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "خروج"}:
            print("شكراً لتواصلك مع AROPE. وداعاً!")
            break

        print("\nالمساعد: ", end="", flush=True)
        # response = chain.invoke(user_input)
        response = chain.invoke(
            user_input,
            config={"callbacks": callbacks}  # ← pass here
        )
        print(response)
        print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embeddings   = build_embeddings()
    vectorstore  = load_vectorstore(embeddings)
    chain        = build_chain(vectorstore)
    chat_loop(chain)
