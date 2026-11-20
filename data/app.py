import streamlit as st
import tempfile
import os

from rag_engine import (
    load_pdf,
    create_chunks,
    build_vectors,
    load_database,
    retrieve,
    ask_rag
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Karnataka Tourism RAG",
    page_icon="🏛️",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    font-size: 38px;
    font-weight: bold;
    color: #176B45;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)


# SIDEBAR
st.sidebar.title("🏛️ Karnataka Tourism")

st.sidebar.caption("Explore Karnataka with AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📌 Project Information",
        "🛠️ Technologies Used",
        "🔄 RAG Pipeline",
        "💬 Tourism Chatbot"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Discover the beauty, culture and heritage of Karnataka."
)


# SESSION STATE
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# PDF UPLOAD
st.sidebar.subheader("📄 Upload PDF")

uploaded_files = st.sidebar.file_uploader(
    "Upload Karnataka Tourism PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# PROCESS PDF
if st.sidebar.button("Build RAG Database"):

    if uploaded_files:

        all_chunks = []

        with st.spinner("Processing PDF documents..."):

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())

                    temp_path = temp_file.name

                try:
                    # 1. Load PDF
                    text = load_pdf(temp_path)

                    # 2. Create Chunks
                    chunks = create_chunks(text)

                    all_chunks.extend(chunks)

                finally:
                    os.remove(temp_path)

            if not all_chunks:
                st.sidebar.error("No readable text found in PDFs.")

            else:
                # 3. Build Vectors
                vectors = build_vectors(all_chunks)

                # 4. Load Database
                index = load_database(vectors)

                st.session_state.index = index
                st.session_state.chunks = all_chunks

                st.sidebar.success(
                    "RAG Database Created Successfully!"
                )

                st.sidebar.info(
                    f"Total Chunks: {len(all_chunks)}"
                )

    else:
        st.sidebar.warning("Please upload PDF files.")


# HOME PAGE
if menu == "🏠 Home":

    st.markdown(
        '<p class="title">🌿 Karnataka Tourism RAG Assistant</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Your AI-powered travel guide to Karnataka</p>',
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1606293926075-69a00dbfde81",
        caption="Discover Karnataka Tourism",
        use_container_width=True
    )

    st.write("""
    Welcome to Karnataka Tourism RAG Assistant!

    Explore:

    - Famous tourist destinations
    - Historical monuments
    - Temples and cultural places
    - Waterfalls and hill stations
    - Food and traditions
    - Travel information
    """)

    st.success("Explore Karnataka with Artificial Intelligence!")


# PROJECT INFORMATION
elif menu == "📌 Project Information":

    st.title("📌 About the Project")

    st.write("""
    **Project Name:** Karnataka Tourism RAG Application

    **Objective:**

    To develop an AI-powered chatbot that provides
    information about Karnataka tourism using PDF documents.

    **Main Features:**

    1. Load PDF documents.
    2. Create text chunks.
    3. Generate embeddings.
    4. Store vectors in database.
    5. Retrieve relevant information.
    6. Generate AI answers.
    """)


# TECHNOLOGIES USED
elif menu == "🛠️ Technologies Used":

    st.title("🛠️ Technologies Used")

    technologies = {
        "Python": "Programming language",
        "Streamlit": "Frontend development",
        "PyPDF": "Extract text from PDF",
        "LangChain": "Text splitting",
        "OpenAI": "AI embeddings and answers",
        "FAISS": "Vector database",
        "NumPy": "Numerical operations"
    }

    for tech, description in technologies.items():

        st.subheader(tech)

        st.write(description)

        st.divider()


# RAG PIPELINE
elif menu == "🔄 RAG Pipeline":

    st.title("🔄 RAG Pipeline Flowchart")

    st.graphviz_chart("""
    digraph RAG {

        rankdir=TB;

        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#DFF3E7",
            color="#176B45"
        ];

        A [label="Tourism PDF Documents"];
        B [label="Load PDF"];
        C [label="Create Chunks"];
        D [label="Build Vectors"];
        E [label="Load Vector Database"];
        F [label="User Question"];
        G [label="Retrieve Information"];
        H [label="Ask RAG / OpenAI"];
        I [label="Final Answer"];

        A -> B -> C -> D -> E;
        E -> G;
        F -> G -> H -> I;
    }
    """)

    st.info(
        "RAG retrieves relevant information from PDF documents "
        "and uses AI to generate answers."
    )


# TOURISM CHATBOT
elif menu == "💬 Tourism Chatbot":

    st.title("💬 Karnataka Tourism AI Chatbot")

    st.caption(
        "Ask questions about Karnataka tourism."
    )

    if st.session_state.index is None:

        st.warning(
            "Please upload PDF files and build the RAG database first."
        )

    else:

        # Display Chat History
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        # User Question
        question = st.chat_input(
            "Ask about Karnataka tourism..."
        )

        if question:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("Searching information..."):

                    try:

                        # 5. Retrieve Information
                        results = retrieve(
                            question,
                            st.session_state.index,
                            st.session_state.chunks
                        )

                        # 6. Ask RAG
                        answer = ask_rag(
                            question,
                            st.session_state.index,
                            st.session_state.chunks
                        )

                        st.markdown(answer)

                        if results:
                            with st.expander("View Retrieved Information"):
                                for result in results:
                                    st.write(result)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })

                    except Exception as e:

                        st.error(f"Error: {e}")

        # Clear Chat
        if st.button("🗑️ Clear Chat"):

            st.session_state.messages = []

            st.rerun()


# FOOTER
st.sidebar.divider()

st.sidebar.caption(
    "Made with ❤️ using Python and Streamlit"
)

st.sidebar.caption(
    "Karnataka Tourism RAG Project"
)
