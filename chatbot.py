"""Chat with a YouTube video using the saved (pickled) RAG components.

No training or re-indexing happens here. This script only LOADS the files that
the notebook saved in saved_files/, then answers your questions.

Usage:
    python chatbot.py                      # interactive mode
    python chatbot.py "What is RAG?"       # ask one question and exit
"""
import sys
from pathlib import Path

import joblib
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

SAVED = Path(__file__).parent / "saved_files"

# 1. Load the saved settings and the saved FAISS index
meta = joblib.load(SAVED / "metadata.pkl")
print(f"Loading saved index for video: {meta['video_id']} ({meta['num_chunks']} chunks)")

# The SAME embedding model that built the index must be used to embed new questions.
embeddings = HuggingFaceEmbeddings(model_name=meta["embedding_model"])

# allow_dangerous_deserialization is needed because index.pkl is a pickle file.
# It is safe here because these are YOUR files. Never load pickle files from strangers.
vector_store = FAISS.load_local(
    str(SAVED / "faiss_index"), embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": meta["k"]})

# 2. Load the language model (downloaded from Hugging Face, not trained here)
pipe = pipeline(
    "text-generation",
    model=meta["llm_model"],
    max_new_tokens=256,
    return_full_text=False,   # return only the answer, not the prompt
)
llm = ChatHuggingFace(
    llm=HuggingFacePipeline(pipeline=pipe),
    model_id=meta["llm_model"],
    tokenizer=pipe.tokenizer,
)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(chain.invoke(" ".join(sys.argv[1:])))
    else:
        print("Ask questions about the video. Type 'quit' to exit.")
        while True:
            q = input("\nYou: ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            print("Bot:", chain.invoke(q))
