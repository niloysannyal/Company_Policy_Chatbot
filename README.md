# 🤖 Company Policy Chatbot (RAG-based)

A **Retrieval-Augmented Generation (RAG)** chatbot that provides quick and accurate answers about **company policies** using a local **Chroma vector database**, **SentenceTransformer embeddings**, and optional integration with **Google Gemini** for natural and contextual responses.

## 🌐 Live Demo

<p align="center">
  <a href="https://company-policy-chatbot.vercel.app/" target="_blank">
    <img width="1920" height="1078" alt="Screenshot (206)" src="https://github.com/user-attachments/assets/ba95150c-7fb9-4cca-b65d-e60c5eccc060" />
    <img src="https://img.shields.io/badge/Live%20Website-Policy%20Assistant-orange?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Live Website Button"/>
  </a>
</p>

---

## 🧠 Overview

The **Company Policy Chatbot** is designed to help employees and HR teams quickly find answers to policy-related questions such as working hours, leave policies, and remote work rules.  
It uses a **Retrieval-Augmented Generation (RAG)** pipeline that retrieves relevant context from a structured company dataset and generates accurate answers — either using a **local summarizer** or **Gemini LLM** if available.

---

## 📂 Folder Structure

```
Company_Policy_Chatbot/
├── dataset/
│ └── dataset.jsonl     # JSONL dataset containing Q/A pairs
├── vector_db/
│ └── chroma.sqlite3    # Persisted Chroma vector database
├── frontend/
│ └── index.html        # Optional web interface
├── .env                # Environment variables (e.g., GEMINI_API_KEY)
├── .gitignore          # Git ignore file
├── main.py             # Entry point or demo script (if any)
├── build_index.py      # Builds the Chroma vector index
├── rag_chatbot.py      # Core RAG chatbot logic
├── main.py             # FastAPI backend API
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 🧾 Sample Dataset (`dataset.jsonl`)

```json
{"messages": [
    {"role": "system", "content": "You are an informative assistant."},
    {"role": "user", "content": "What is the company’s leave policy?"},
    {"role": "assistant", "content": "Employees are entitled to annual, sick, and casual leave as per company guidelines. All leave requests must be submitted through the HR portal or to the reporting manager at least three days in advance, except in emergencies. Unused annual leave may be carried over up to the maximum limit set by HR."}
]}

{"messages": [
    {"role": "system", "content": "You are an informative assistant."},
    {"role": "user", "content": "How can employees request remote work?"},
    {"role": "assistant", "content": "Employees can request remote work by submitting a formal request through the HR or attendance system. Approval is subject to the manager’s discretion based on project requirements, role suitability, and performance history."}
]}

{"messages": [
    {"role": "system", "content": "You are an informative assistant."},
    {"role": "user", "content": "What are the company’s regular working hours?"},
    {"role": "assistant", "content": "Standard working hours are from 9:00 AM to 6:00 PM, Monday through Friday, with a one-hour lunch break. Employees are expected to maintain punctuality unless flexible work hours have been approved."}
]}
```
---

## 🚀 Features

✅ **RAG Pipeline:** Retrieve relevant text chunks and generate accurate answers.  
✅ **Chroma Vector Store:** Efficient, persistent storage for embeddings.  
✅ **SentenceTransformer Embeddings:** Uses `all-MiniLM-L6-v2` for semantic similarity.  
✅ **MMR Reranking:** Ensures diverse and relevant retrieval results.  
✅ **Gemini LLM Integration (optional):** For more fluent, context-aware answers.  
✅ **FastAPI Backend:** Clean REST API interface (`/ask` endpoint).  
✅ **CORS Enabled:** Frontend and backend communicate seamlessly.  
✅ **Local Summarizer Fallback:** Works even without Gemini API key.  

---

## 🧰 Tech Stack

- **Python 3.10+**
- **FastAPI**
- **ChromaDB**
- **SentenceTransformers**
- **dotenv**
- **NumPy**
- **tqdm**
- **Google Generative AI (Gemini)**

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Company_Policy_Chatbot.git
cd Company_Policy_Chatbot
```
### #️⃣ Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # mac or linux 
venv\Scripts\activate      # Windows
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Add Your Gemini API Key
**Create a .env file in the project root:**
```
GEMINI_API_KEY=your_api_key_here
```
### 5️⃣ Build the Vector Index
```
python build_index.py
```
**This script:**
- Reads your dataset `(dataset/dataset.jsonl)`
- Splits answers into overlapping chunks 
- Embeds them using SentenceTransformers 
- Stores embeddings into `vector_db/`

### 6️⃣ Run the API Server
```
python main.py              # IDE
uvicorn main:app --reload   # Terminal
```
**Server will start at:**
```aiignore
http://127.0.0.1:8000
```

## 🧩 API Usage

### ▶️ Endpoint: `/ask`

**Method:** `POST`

---

### 📨 Request Body
```
{
  "question": "What is the company’s leave policy?"
}
```
### 📦 Response
```
{
  "answer": "Employees are entitled to annual, sick, and casual leave as per company guidelines. All leave requests must be submitted through the HR portal or to the reporting manager at least three days in advance, except in emergencies. Unused annual leave may be carried over up to the maximum limit set by HR."
}
```
---

## 💻 Frontend Integration

**Find this line in index.html, at the beginning of the script section**
```aiignore
const API_URL = "http://localhost:8000/ask";
```
**Then update this link to your the api endpoint link**

## 🧪 Run Locally in Terminal
**You can also run the chatbot interactively:**
```aiignore
python rag_chatbot.py
```
**Example Output:**
```aiignore
🤖 Company Policy Chatbot (RAG + Chroma)
Type 'exit' to quit.

Ask: What are the company’s working hours?
💬 Answer:
Standard working hours are from 9:00 AM to 6:00 PM, Monday through Friday, with a one-hour lunch break. Employees are expected to maintain punctuality unless flexible work hours have been approved.
```
---

## 🧠 How It Works (RAG Flow)

1. **User Query →** Embed using `SentenceTransformer`.  
2. **Dense Retrieval →** Query Chroma vector store for top matches.  
3. **MMR Reranking →** Select the most relevant & diverse chunks.  
4. **Context Construction →** Combine retrieved chunks into a context block.  
5. **Answer Generation →**  
   - If **Gemini API** is available → Generate fluent, context-aware answer.  
   - Else → Use local heuristic summarizer.  
6. **Return Answer →** With relevant context and citations.

---

## 📘 Example Prompts

| Question | Answer |
|-----------|---------|
| What is the company’s leave policy? | Employees are entitled to annual, sick, and casual leave... |
| How can employees request remote work? | Employees can request remote work by submitting a formal request... |
| What are the company’s regular working hours? | Standard working hours are from 9:00 AM to 6:00 PM... |

---

## 🛠️ Future Enhancements

- 🔐 Add authentication for internal HR access  
- 📚 Expand dataset with detailed HR, IT, and Admin policies  
- 🎨 Improve frontend UI for better user experience  
- 💬 Integrate chatbot with Slack or Microsoft Teams for internal use  
- 🌐 Add multilingual support for diverse employees  
- 🤖 Enhance retrieval accuracy using vector embeddings and fine-tuned LLM  

---

## 🧑‍💻 Author

**Developed by:** Niloy Sannyal  
**GitHub:** [@niloysannyal](https://github.com/niloysannyal)  
**Email:** niloysannyal@gmail.com  
**Location:** Dhaka, Bangladesh  

---

## 🪪 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it for both personal and commercial purposes.  

---

## ⭐ Support

If you find this project helpful, please consider giving it a ⭐ on [GitHub](https://github.com/niloysannyal)!  
Your support helps others discover this open-source HR Policy Chatbot project. 💬
