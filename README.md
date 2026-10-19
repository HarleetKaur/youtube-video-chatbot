# 🎬 YouTube Video Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a YouTube video using its transcript.

## How it works
1. Fetch the transcript with `youtube-transcript-api`
2. Split it into overlapping chunks
3. Convert chunks to vectors with a multilingual Sentence-Transformers model
4. Store the vectors in a FAISS index
5. For each question, retrieve the most similar chunks and give them to `Qwen2.5-1.5B-Instruct` to write the answer

## Project structure
```
.
├── Youtubechatbot.ipynb     # Notebook that builds everything and saves the files below
├── chatbot.py               # Loads the saved files and answers questions (no rebuilding)
├── requirements.txt         # Python packages (versions match the notebook)
├── saved_files/
│   ├── faiss_index/
│   │   ├── index.faiss      # vector index
│   │   └── index.pkl        # text of chunks + ids (pickle)
│   ├── chunks.pkl           # list of transcript chunks (pickle)
│   └── metadata.pkl         # model names and settings used to build the index
└── README.md
```

## Run it yourself (no retraining)
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python chatbot.py "Is the topic of aliens discussed in this video?"
```
Run `python chatbot.py` with no question for interactive mode.

The first run downloads two pre-trained models from Hugging Face (the embedding model and `Qwen2.5-1.5B-Instruct`, about 4 GB in total). Nothing is trained or re-indexed.

## Use a different video
Open `Youtubechatbot.ipynb` in Google Colab, change `video_id`, run all cells, and download the new `saved_files`.

## Notes
- `index.pkl` and the other `.pkl` files are pickle files. Only load pickle files you trust.
- Answers run on CPU and can take a minute or two. A small model can make mistakes, so always check important answers in the video.
