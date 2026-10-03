import streamlit as st

from backend.document_processor import (
    extract_text_from_pdf,
    create_chunks
)

from backend.embeddings import EmbeddingModel
from backend.vector_store import VectorStore
from backend.rag import ask_llm


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def load_vector_store():
    return VectorStore()


embedding_model = load_embedding_model()
vector_store = load_vector_store()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎓 AI Academic Assistant")

st.write(
    "Upload your academic material and use AI for "
    "summaries, explanations, questions, exams and revision."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("👨‍🎓 Student Profile")

branch = st.sidebar.selectbox(
    "Select Branch",
    [
        "CSE",
        "CSM",
        "AI & DS",
        "ECE",
        "EEE",
        "Mechanical",
        "Civil",
        "Other"
    ]
)

semester = st.sidebar.selectbox(
    "Select Semester",
    [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8"
    ]
)

subject = st.sidebar.text_input(
    "Enter Subject",
    "Data Structures"
)


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.title("🛠️ AI Features")

feature = st.sidebar.radio(
    "Choose an option",
    [
        "📖 Read Material",
        "📝 Summarize",
        "💡 Explain a Topic",
        "❓ Ask Questions",
        "📝 Generate Exam Answer",
        "🧠 Generate Quiz",
        "⚡ Quick Revision",
        "🗂️ Generate Flashcards",
        "🔑 Important Questions"
    ]
)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.header("📄 Upload Academic Material")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file:

    st.success(
        f"{uploaded_file.name} uploaded successfully!"
    )

    if st.button("🧠 Process & Index PDF"):

        with st.spinner(
            "Processing your PDF..."
        ):

            pages = extract_text_from_pdf(
                uploaded_file
            )

            if not pages:

                st.error(
                    "No readable text was found in the PDF."
                )

                st.stop()

            chunks = create_chunks(
                pages,
                chunk_size=1000,
                overlap=150
            )

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = (
                embedding_model
                .create_embeddings(texts)
            )

            vector_store.add_documents(
                chunks,
                embeddings,
                branch,
                semester,
                subject,
                uploaded_file.name
            )

        st.success(
            f"PDF processed successfully! "
            f"{len(pages)} pages and "
            f"{len(chunks)} chunks indexed."
        )


# --------------------------------------------------
# CHECK PDF
# --------------------------------------------------

if not uploaded_file:

    st.info(
        "Upload a PDF to start using the AI assistant."
    )

    st.stop()


# --------------------------------------------------
# READ MATERIAL
# --------------------------------------------------

if feature == "📖 Read Material":

    st.header("📖 Read Material")

    pages = extract_text_from_pdf(
        uploaded_file
    )

    page_number = st.number_input(
        "Select page",
        min_value=1,
        max_value=len(pages),
        value=1
    )

    selected_page = pages[
        page_number - 1
    ]

    st.subheader(
        f"Page {selected_page['page']}"
    )

    st.write(
        selected_page["text"]
    )


# --------------------------------------------------
# RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve_context(query, top_k=5):

    query_embedding = (
        embedding_model
        .create_embeddings([query])[0]
    )

    results = vector_store.search(
        query_embedding,
        branch,
        semester,
        subject,
        top_k=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    if not documents:

        return "", []

    context_parts = []

    pages = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        context_parts.append(
            document
        )

        pages.append(
            metadata["page"]
        )

    context = "\n\n".join(
        context_parts
    )

    return context, pages


# --------------------------------------------------
# SUMMARIZE
# --------------------------------------------------

if feature == "📝 Summarize":

    st.header("📝 Summarize")

    topic = st.text_input(
        "What topic do you want to summarize?",
        "the main concepts"
    )

    if st.button("✨ Generate Summary"):

        with st.spinner(
            "Finding relevant material..."
        ):

            context, pages = retrieve_context(
                topic,
                top_k=3
            )

        if not context:

            st.error(
                "No relevant material found. "
                "Please click 'Process & Index PDF' first."
            )

        else:

            with st.spinner(
                "Generating summary..."
            ):

                answer = ask_llm(
                    f"""
Summarize the following topic:

{topic}

Give:
1. Main definition
2. Important concepts
3. Key points
4. Simple explanation
""",
                    context
                )

            st.subheader("📌 Summary")

            st.write(answer)

            st.caption(
                f"Source pages: {sorted(set(pages))}"
            )


# --------------------------------------------------
# EXPLAIN TOPIC
# --------------------------------------------------

elif feature == "💡 Explain a Topic":

    st.header("💡 Explain a Topic")

    topic = st.text_input(
        "Enter the topic you want explained"
    )

    if st.button("💡 Explain"):

        if not topic:

            st.warning(
                "Please enter a topic."
            )

        else:

            with st.spinner(
                "Finding relevant material..."
            ):

                context, pages = retrieve_context(
                    topic,
                    top_k=3
                )

            if not context:

                st.error(
                    "No relevant information found."
                )

            else:

                with st.spinner(
                    "Generating explanation..."
                ):

                    answer = ask_llm(
                        f"""
Explain this topic clearly:

{topic}

Explain it like a teacher teaching
a college student.

Include a simple example if possible.
""",
                        context,
                        max_tokens=600
                    )

                st.subheader(
                    "💡 Explanation"
                )

                st.write(answer)

                st.caption(
                    f"Source pages: {sorted(set(pages))}"
                )


# --------------------------------------------------
# ASK QUESTIONS
# --------------------------------------------------

elif feature == "❓ Ask Questions":

    st.header(
        "❓ Ask Your Academic Assistant"
    )

    question = st.text_area(
        "Enter your question"
    )

    if st.button("🚀 Ask AI"):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching your material..."
            ):

                context, pages = retrieve_context(
                    question,
                    top_k=3
                )

            if not context:

                st.error(
                    "No relevant information found. "
                    "Please process the PDF first."
                )

            else:

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = ask_llm(
                        question,
                        context
                    )

                st.subheader(
                    "🤖 Answer"
                )

                st.write(answer)

                st.caption(
                    f"📄 Source pages: "
                    f"{sorted(set(pages))}"
                )


# --------------------------------------------------
# EXAM ANSWER
# --------------------------------------------------

elif feature == "📝 Generate Exam Answer":

    st.header(
        "📝 Generate Exam Answer"
    )

    question = st.text_area(
        "Enter the exam question"
    )

    marks = st.selectbox(
        "Select marks",
        [2, 5, 7, 10, 14]
    )

    if st.button("📝 Generate Answer"):

        if not question:

            st.warning(
                "Please enter the question."
            )

        else:

            with st.spinner(
                "Finding relevant content..."
            ):

                context, pages = retrieve_context(
                    question,
                    top_k=3
                )

            with st.spinner(
                "Preparing exam answer..."
            ):

                answer = ask_llm(
    f"""
Generate a {marks}-mark exam answer
for this question:

{question}

Format the answer appropriately for
a college examination.

Include headings, explanations,
examples and important points.
""",
    context,
    max_tokens=500
)

            st.subheader(
                f"📝 {marks}-Mark Answer"
            )

            st.write(answer)


# --------------------------------------------------
# QUIZ
# --------------------------------------------------

elif feature == "🧠 Generate Quiz":

    st.header(
        "🧠 Generate Quiz"
    )

    topic = st.text_input(
        "Enter quiz topic",
        "important concepts"
    )

    number = st.selectbox(
        "Number of questions",
        [5, 10, 15]
    )

    if st.button("🧠 Generate Quiz"):

        with st.spinner(
            "Preparing quiz..."
        ):

            context, pages = retrieve_context(
                topic,
                top_k=3
            )

            answer = ask_llm(
                f"""
Create a {number}-question quiz
from this academic material.

Use multiple-choice questions.

For each question provide:
A, B, C, D options.

At the end provide the answer key.

Topic:
{topic}
""",
                context
            )

        st.write(answer)


# --------------------------------------------------
# QUICK REVISION
# --------------------------------------------------

elif feature == "⚡ Quick Revision":

    st.header(
        "⚡ Quick Revision"
    )

    topic = st.text_input(
        "Enter topic for revision",
        "important concepts"
    )

    if st.button("⚡ Start Revision"):

        with st.spinner(
            "Preparing revision notes..."
        ):

            context, pages = retrieve_context(
                topic,
                top_k=3
            )

            answer = ask_llm(
                f"""
Create quick revision notes for:

{topic}

Include:
• Definitions
• Important points
• Formulas if available
• Examples
• Things to remember

Keep it concise.
""",
                context
            )

        st.write(answer)


# --------------------------------------------------
# FLASHCARDS
# --------------------------------------------------

elif feature == "🗂️ Generate Flashcards":

    st.header(
        "🗂️ Generate Flashcards"
    )

    topic = st.text_input(
        "Enter flashcard topic",
        "important concepts"
    )

    if st.button("🗂️ Create Flashcards"):

        with st.spinner(
            "Creating flashcards..."
        ):

            context, pages = retrieve_context(
                topic,
                top_k=3
            )

            answer = ask_llm(
    f"""
Create exactly 10 study flashcards from this topic:

{topic}

Each flashcard must contain:
Question:
Answer:

Number them CARD 1 through CARD 10.

Do not add an introduction.
Do not stop before CARD 10.
""",
    context,
    max_tokens=600
)

        st.write(answer)


# --------------------------------------------------
# IMPORTANT QUESTIONS
# --------------------------------------------------

elif feature == "🔑 Important Questions":

    st.header(
        "🔑 Important Questions"
    )

    topic = st.text_input(
        "Enter topic",
        "important topics"
    )

    if st.button("🔑 Generate Questions"):

        with st.spinner(
            "Finding important questions..."
        ):

            context, pages = retrieve_context(
                topic,
                top_k=3
            )

            answer = ask_llm(
                f"""
Generate important examination questions
from this academic material.

Include:
• 2-mark questions
• 5-mark questions
• 7-mark questions
• 10/14-mark questions

Topic:
{topic}
""",
                context
            )

        st.write(answer)