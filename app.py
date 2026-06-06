import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2
import io
import re
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=os.getenv("IBM_API_KEY")
)
project_id = os.getenv("IBM_PROJECT_ID")

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    credentials=credentials,
    project_id=project_id,
    params={
        "max_new_tokens": 800,
        "temperature": 0.7,
        "repetition_penalty": 1.1
    }
)

# -------------------------------------------------------------
# REAL RAG — FAISS + HuggingFace Embeddings
# -------------------------------------------------------------
INTERVIEW_DOCUMENTS = [
    "Software Engineer interview requires System Design, Data Structures, REST API design, CI/CD pipelines. HR Focus: Agility and collaborative problem solving.",
    "Data Scientist interview requires Machine Learning, statistical modeling, feature engineering, SQL. HR Focus: Communication of complex insights.",
    "Product Manager interview requires Product metrics, roadmap prioritization, user stories, conflict resolution. HR Focus: Leadership and strategic vision.",
    "Web Developer interview requires HTML, CSS, JavaScript, React, REST APIs, responsive design. HR Focus: Creativity and deadline management.",
    "DevOps Engineer interview requires Docker, Kubernetes, CI/CD, AWS, Azure, monitoring tools. HR Focus: Reliability and automation mindset.",
    "Machine Learning Engineer interview requires Deep learning, model deployment, MLOps, Python, cloud ML. HR Focus: Research mindset and scalability.",
    "Business Analyst interview requires Requirements gathering, process mapping, SQL, data visualization. HR Focus: Stakeholder management and analytical thinking.",
    "AI and ML interview requires Deep learning, NLP, Computer Vision, TensorFlow, PyTorch, model deployment. HR Focus: Research mindset and experimentation.",
    "Behavioral interview questions use STAR method: Situation, Task, Action, Result format for answering.",
    "Technical interview preparation includes practicing coding problems, system design, and algorithm questions.",
    "HR interview tips: Research company, prepare questions, dress professionally, arrive early, follow up after interview.",
    "Resume tips: Keep it one page, use action verbs, quantify achievements, tailor to job description.",
    "Salary negotiation: Research market rate, know your value, negotiate benefits, be professional and confident.",
    "Common interview mistakes: Not researching company, being unprepared, negative talk about past employers.",
    "Body language tips: Maintain eye contact, firm handshake, sit straight, smile, listen actively.",
    "Python interview: Focus on data structures, OOP, list comprehensions, decorators, generators, pandas, numpy.",
    "Deep Learning interview: Neural networks, CNN, RNN, LSTM, transformers, backpropagation, optimization.",
    "NLP interview: Tokenization, embeddings, BERT, GPT, sentiment analysis, text classification.",
    "Data Analysis interview: Pandas, NumPy, data cleaning, EDA, visualization with matplotlib and seaborn.",
    "Fresher interview tips: Highlight projects, internships, academic achievements, willingness to learn.",
]

@st.cache_resource
def setup_faiss():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(INTERVIEW_DOCUMENTS)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def retrieve_rag_context(query: str, n_results: int = 3) -> str:
    try:
        vectorstore = setup_faiss()
        docs = vectorstore.similarity_search(query, k=n_results)
        context = "\n".join([doc.page_content for doc in docs])
        return f"[RAG Retrieved Context]:\n{context}"
    except Exception as e:
        return "[RAG Context]: Standard interview frameworks apply."

# -------------------------------------------------------------
# Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Trainer Agent",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.main-header h1 { font-size: 2.2rem; margin-bottom: 5px; }
.main-header p { font-size: 1rem; opacity: 0.85; }

.rag-badge {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 5px 12px;
    border-radius: 20px;
    color: white;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin: 5px;
}

.stat-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 3px solid #667eea;
}
.stat-number { font-size: 1.5rem; font-weight: 700; color: #667eea; }
.stat-label { font-size: 0.8rem; color: #888; margin-top: 4px; }

.question-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px 25px;
    border-radius: 15px;
    color: white;
    font-size: 1.05rem;
    font-weight: 500;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}

.feedback-card {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 15px 20px;
    border-radius: 12px;
    color: #1a1a2e;
    margin: 10px 0;
    font-weight: 600;
}

.tip-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 15px 20px;
    border-radius: 12px;
    color: #1a1a2e;
    margin: 8px 0;
    font-weight: 500;
}

.resource-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-left: 4px solid #667eea;
    margin: 10px 0;
}

.resume-badge {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 8px 15px;
    border-radius: 20px;
    color: #1a1a2e;
    font-weight: 600;
    font-size: 0.85rem;
    text-align: center;
    margin: 8px 0;
}

.complete-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 15px 0;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎯 Interview Trainer Agent</h1>
    <p>Powered by IBM Granite AI | Real RAG with FAISS Vector Search</p>
    <p><small>Problem Statement No.22 | Edunet Foundation</small></p>
    <div>
        <span class="rag-badge">🔍 FAISS RAG</span>
        <span class="rag-badge">🤖 IBM Granite</span>
        <span class="rag-badge">📊 Vector Search</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Session State
for key, default in {
    "messages": [],
    "profile_name": "",
    "job_role": "",
    "exp_level": "Fresher",
    "resume_text": "",
    "score": 0.0,
    "attempts": 0,
    "current_question": "",
    "profile_saved": False,
    "practice_history": [],
    "practice_started": False,
    "total_questions": 5,
    "practice_role": "",
    "p_question_type": "Technical",
    "p_difficulty": "Easy",
    "show_feedback": False,
    "last_feedback": "",
    "last_score": 0.0,
    "saved_history": [],
    "tab3_tips": "",
    "tab3_category": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def trigger_agent_generation(user_prompt: str):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    retrieved_context = retrieve_rag_context(
        f"{user_prompt} {st.session_state.job_role} {st.session_state.exp_level}"
    )

    system_instruction = f"""You are an expert AI Interview Trainer Agent powered by Real RAG (FAISS Vector Search).

--- USER PROFILE ---
Candidate Name: {st.session_state.profile_name if st.session_state.profile_name else 'Candidate'}
Target Job Role: {st.session_state.job_role if st.session_state.job_role else 'General'}
Experience Level: {st.session_state.exp_level}
Resume: {st.session_state.resume_text[:800] if st.session_state.resume_text else 'Not provided.'}

--- REAL RAG RETRIEVED CONTEXT (FAISS Vector Search) ---
{retrieved_context}

--- INSTRUCTIONS ---
1. Use RAG context and resume to personalize responses.
2. Give SHORT and SIMPLE answers.
3. Use simple English and bullet points.
4. For STAR answers: S/T/A/R each 1-2 lines only."""

    full_prompt = f"{system_instruction}\n\nUser: {user_prompt}\n\nAssistant:"

    with st.spinner("🤖 IBM Granite + FAISS RAG analyzing..."):
        try:
            ai_response = model.generate_text(prompt=full_prompt)
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response
            })
        except Exception as e:
            st.error(f"❌ Error: {e}")

# SIDEBAR
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=70)
    st.markdown("---")

    st.markdown("### 🔍 RAG Status")
    try:
        vs = setup_faiss()
        st.success(f"✅ FAISS RAG Ready! {len(INTERVIEW_DOCUMENTS)} docs indexed")
    except Exception as e:
        st.error(f"❌ RAG Error: {e}")

    st.markdown("---")
    st.header("👤 Profile Setup")

    profile_name = st.text_input("Your Name:", placeholder="Enter your name")
    job_role = st.text_input("Target Job Role:", placeholder="Ex: Software Engineer")
    exp_level = st.selectbox("Experience Level:", [
        "Fresher", "Mid-Level (2-5 Years)", "Senior (5+ Years)"
    ])

    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("PDF or TXT", type=["txt", "pdf"])

    if uploaded_file is not None:
        st.markdown('<div class="resume-badge">📄 Resume Ready!</div>', unsafe_allow_html=True)

    if st.button("💾 Save Profile", type="primary", use_container_width=True):
        st.session_state.profile_name = profile_name
        st.session_state.job_role = job_role
        st.session_state.exp_level = exp_level
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                resume_text = ""
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text
                st.session_state.resume_text = resume_text
            else:
                st.session_state.resume_text = uploaded_file.read().decode("utf-8")
        st.session_state.profile_saved = True
        st.success(f"✅ Welcome {profile_name}!")
        st.rerun()

    if st.session_state.profile_saved:
        st.markdown(f'<div class="resume-badge">✅ {st.session_state.profile_name} | {st.session_state.job_role}</div>', unsafe_allow_html=True)
        if st.session_state.resume_text:
            st.markdown('<div class="resume-badge">📄 Resume Loaded!</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Performance")
    avg = st.session_state.score / st.session_state.attempts if st.session_state.attempts > 0 else 0
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Score", f"{avg:.1f}/10")
    with col2:
        st.metric("Attempts", st.session_state.attempts)

    if st.button("🔄 Reset Score", use_container_width=True):
        st.session_state.score = 0.0
        st.session_state.attempts = 0
        st.session_state.practice_history = []
        st.session_state.saved_history = []
        st.session_state.practice_started = False
        st.session_state.current_question = ""
        st.rerun()

    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    if st.button("📝 Generate 5 Questions", use_container_width=True):
        trigger_agent_generation(
            f"Generate 5 interview questions with simple answers for {st.session_state.job_role} at {st.session_state.exp_level} level."
        )
        st.rerun()

    if st.button("💼 HR Behavioral Questions", use_container_width=True):
        trigger_agent_generation(
            f"Give 5 HR behavioral questions with short STAR method answers for {st.session_state.job_role}."
        )
        st.rerun()

    if st.button("🔧 Technical Questions", use_container_width=True):
        trigger_agent_generation(
            f"Give 5 technical interview questions with simple answers for {st.session_state.job_role} at {st.session_state.exp_level} level."
        )
        st.rerun()

    if st.button("💡 Interview Tips", use_container_width=True):
        trigger_agent_generation(
            f"Give top 10 simple interview tips for {st.session_state.job_role} role."
        )
        st.rerun()

    if st.button("📄 Resume Based Questions", use_container_width=True):
        if st.session_state.resume_text:
            trigger_agent_generation(
                f"Based on my resume, generate 5 interview questions for {st.session_state.job_role} role."
            )
        else:
            st.warning("⚠️ Please upload resume first!")
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# MAIN TABS
tab1, tab2, tab3 = st.tabs(["💬 Chat", "✍️ Practice Mode", "📚 Tips & Resources"])

# TAB 1
with tab1:
    if not st.session_state.job_role:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#667eea22,#764ba222);
                    padding:30px;border-radius:15px;text-align:center;">
            <h3>👋 Welcome to Interview Trainer Agent!</h3>
            <p>Set your <b>Profile</b> and <b>Job Role</b> in the sidebar to begin!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        avg = st.session_state.score / st.session_state.attempts if st.session_state.attempts > 0 else 0
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">🎯</div><div class="stat-label">{st.session_state.job_role}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.exp_level.split()[0]}</div><div class="stat-label">Experience</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{avg:.1f}</div><div class="stat-label">Avg Score</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.attempts}</div><div class="stat-label">Attempts</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about interview preparation..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        if st.session_state.resume_text:
            enhanced_prompt = f"""{prompt}
[My Resume]:
{st.session_state.resume_text[:800]}
Answer based on my resume."""
        else:
            enhanced_prompt = prompt
        trigger_agent_generation(enhanced_prompt)
        st.rerun()

# TAB 2
with tab2:
    st.subheader("✍️ Mock Interview Practice")

    if not st.session_state.practice_started:
        st.write("⚙️ Setup your practice session:")

        col1, col2, col3 = st.columns(3)
        with col1:
            practice_role = st.text_input(
                "🎯 Role:",
                value=st.session_state.job_role if st.session_state.job_role else "",
                placeholder="Ex: Software Engineer"
            )
        with col2:
            question_type = st.selectbox("📂 Question Type:", [
                "Technical", "Behavioral", "HR", "System Design"
            ])
        with col3:
            difficulty = st.selectbox("📊 Difficulty:", ["Easy", "Medium", "Hard"])

        total_q = st.number_input(
            "📝 How many questions?",
            min_value=1, max_value=20, value=5, step=1
        )

        if st.button("🚀 Start Practice Session", use_container_width=True, type="primary"):
            if practice_role:
                st.session_state.practice_started = True
                st.session_state.practice_history = []
                st.session_state.current_question = ""
                st.session_state.total_questions = int(total_q)
                st.session_state.practice_role = practice_role
                st.session_state.p_question_type = question_type
                st.session_state.p_difficulty = difficulty
                st.session_state.show_feedback = False
                st.session_state.last_feedback = ""
                st.rerun()
            else:
                st.warning("⚠️ Please enter a Job Role!")

        if st.session_state.saved_history:
            st.markdown("---")
            st.subheader(f"📋 Previous Session — {len(st.session_state.saved_history)} Questions")
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Delete History", use_container_width=True):
                    st.session_state.saved_history = []
                    st.rerun()
            total_saved = len(st.session_state.saved_history)
            avg_saved = sum(i['score'] for i in st.session_state.saved_history) / total_saved if total_saved > 0 else 0
            st.markdown(f"⭐ **Avg Score: {avg_saved:.1f}/10** | 📊 **Total: {total_saved} Questions**")
            for i, item in enumerate(reversed(st.session_state.saved_history)):
                q_num = total_saved - i
                with st.expander(f"Q{q_num}: {item['question'][:55]}... | ⭐ {item['score']}/10"):
                    st.markdown(f"**🎤 Question:** {item['question']}")
                    st.markdown(f"**✍️ Your Answer:** {item['answer']}")
                    st.markdown('<div class="feedback-card">📊 AI Feedback</div>', unsafe_allow_html=True)
                    st.markdown(item['feedback'])

    else:
        done = len(st.session_state.practice_history)
        total = st.session_state.total_questions

        st.progress(done / total if total > 0 else 0)
        st.markdown(f"**📊 Progress: {done} / {total} Questions Done**")
        st.markdown(f"🎯 `{st.session_state.practice_role}` | 📂 `{st.session_state.p_question_type}` | 📊 `{st.session_state.p_difficulty}`")

        if done >= total:
            avg = st.session_state.score / st.session_state.attempts if st.session_state.attempts > 0 else 0
            st.markdown(f"""
            <div class="complete-card">
                <h2>🎉 Practice Session Complete!</h2>
                <p>✅ Total: {total} Questions</p>
                <p>⭐ Average Score: {avg:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

            if st.button("🔄 Start New Session", use_container_width=True, type="primary"):
                st.session_state.saved_history = st.session_state.practice_history.copy()
                st.session_state.practice_started = False
                st.session_state.practice_history = []
                st.session_state.current_question = ""
                st.session_state.show_feedback = False
                st.rerun()

        else:
            if st.session_state.show_feedback and st.session_state.last_feedback:
                st.markdown('<div class="feedback-card">📊 <b>AI Feedback:</b></div>', unsafe_allow_html=True)
                st.markdown(st.session_state.last_feedback)
                st.markdown(f"⭐ **Score: {st.session_state.last_score}/10**")
                st.markdown("---")
                if st.button("➡️ Next Question", use_container_width=True, type="primary"):
                    st.session_state.show_feedback = False
                    st.session_state.last_feedback = ""
                    st.session_state.current_question = ""
                    st.rerun()

            else:
                if not st.session_state.current_question:
                    with st.spinner(f"🤖 FAISS RAG + IBM Granite generating Q{done+1}..."):
                        try:
                            rag_context = retrieve_rag_context(
                                f"{st.session_state.p_question_type} interview {st.session_state.practice_role}"
                            )
                            resume_hint = f"Resume: {st.session_state.resume_text[:400]}" if st.session_state.resume_text else ""
                            prev_questions = ""
                            if st.session_state.practice_history:
                                prev_questions = "Do NOT repeat:\n"
                                for item in st.session_state.practice_history:
                                    prev_questions += f"- {item['question']}\n"

                            q_prompt = f"""{rag_context}
{resume_hint}
{prev_questions}

Generate ONE {st.session_state.p_question_type} question for {st.session_state.practice_role} at {st.session_state.p_difficulty} level.
Return ONLY the question."""
                            question = model.generate_text(prompt=q_prompt)
                            st.session_state.current_question = question.strip()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

                if st.session_state.current_question:
                    st.markdown(f"""
                    <div class="question-card">
                        🎤 <b>Question {done+1} of {total}:</b><br><br>
                        {st.session_state.current_question}
                    </div>
                    """, unsafe_allow_html=True)

                    user_answer = st.text_area(
                        "✍️ Your Answer:",
                        height=150,
                        placeholder="Type your answer here...",
                        key=f"ans_{done}"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Submit Answer", use_container_width=True, type="primary"):
                            if user_answer.strip():
                                with st.spinner("🔍 Evaluating..."):
                                    try:
                                        eval_prompt = f"""Strict interview evaluator.

Question: {st.session_state.current_question}
Answer: {user_answer}

EXACT format:
SCORE: [number]/10
STRENGTHS:
- Point 1
- Point 2
IMPROVEMENTS:
- Point 1
- Point 2
MODEL ANSWER:
Brief ideal answer.

No extra text."""
                                        feedback = model.generate_text(prompt=eval_prompt)
                                        parsed_score = 7.0
                                        try:
                                            m = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10', feedback, re.IGNORECASE)
                                            if m:
                                                parsed_score = float(m.group(1))
                                        except:
                                            pass

                                        st.session_state.practice_history.append({
                                            "question": st.session_state.current_question,
                                            "answer": user_answer,
                                            "feedback": feedback,
                                            "score": parsed_score
                                        })
                                        st.session_state.score += parsed_score
                                        st.session_state.attempts += 1
                                        st.session_state.last_feedback = feedback
                                        st.session_state.last_score = parsed_score
                                        st.session_state.show_feedback = True
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")
                            else:
                                st.warning("⚠️ Write your answer!")

                    with col2:
                        if st.button("⏭️ Skip", use_container_width=True):
                            st.session_state.practice_history.append({
                                "question": st.session_state.current_question,
                                "answer": "Skipped",
                                "feedback": "Skipped.",
                                "score": 0.0
                            })
                            st.session_state.attempts += 1
                            st.session_state.current_question = ""
                            st.session_state.show_feedback = False
                            st.rerun()

        if st.session_state.practice_history:
            st.markdown("---")
            st.subheader(f"📋 Current Session — {len(st.session_state.practice_history)} Done")
            for i, item in enumerate(reversed(st.session_state.practice_history)):
                q_num = len(st.session_state.practice_history) - i
                with st.expander(f"Q{q_num}: {item['question'][:55]}... | ⭐ {item['score']}/10"):
                    st.markdown(f"**🎤 Q:** {item['question']}")
                    st.markdown(f"**✍️ A:** {item['answer']}")
                    st.markdown('<div class="feedback-card">📊 Feedback</div>', unsafe_allow_html=True)
                    st.markdown(item['feedback'])

        st.markdown("---")
        if st.button("🛑 Stop Session", use_container_width=True):
            if st.session_state.practice_history:
                st.session_state.saved_history = st.session_state.practice_history.copy()
            st.session_state.practice_started = False
            st.session_state.current_question = ""
            st.session_state.show_feedback = False
            st.rerun()

# TAB 3
with tab3:
    st.subheader("📚 Interview Tips & Resources")

    col1, col2 = st.columns(2)
    with col1:
        tip_category = st.selectbox("📌 Select Topic:", [
            "General Interview Tips",
            "Technical Interview Preparation",
            "Behavioral Questions (STAR Method)",
            "Body Language & Communication",
            "Salary Negotiation Tips",
            "Questions to Ask the Interviewer",
            "Resume Tips",
            "Common Mistakes to Avoid"
        ])
    with col2:
        tip_role = st.text_input(
            "🎯 For Role:",
            value=st.session_state.job_role if st.session_state.job_role else "",
            placeholder="Ex: Software Engineer"
        )

    if st.button("💡 Get Tips", use_container_width=True, type="primary"):
        with st.spinner("📖 FAISS searching + IBM Granite generating..."):
            try:
                rag_context = retrieve_rag_context(f"{tip_category} {tip_role}")
                tips_prompt = f"""{rag_context}

Simple guide on: {tip_category}
For: {tip_role if tip_role else 'General'}
Numbered list. Short simple points. Max 8."""
                tips = model.generate_text(prompt=tips_prompt)
                st.session_state.tab3_tips = tips
                st.session_state.tab3_category = tip_category
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.tab3_tips:
        st.markdown(f'<div class="tip-card">💡 <b>{st.session_state.tab3_category}</b></div>', unsafe_allow_html=True)
        st.markdown(st.session_state.tab3_tips)
        if st.button("🗑️ Clear Tips"):
            st.session_state.tab3_tips = ""
            st.rerun()

    st.markdown("---")
    st.markdown("### 🌐 Useful Resources")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="resource-card">
            <h4>📖 Practice Platforms</h4>
            <a href="https://leetcode.com" target="_blank">🔗 LeetCode</a><br>
            <a href="https://hackerrank.com" target="_blank">🔗 HackerRank</a><br>
            <a href="https://interviewbit.com" target="_blank">🔗 InterviewBit</a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="resource-card">
            <h4>💼 Job Portals</h4>
            <a href="https://linkedin.com" target="_blank">🔗 LinkedIn</a><br>
            <a href="https://naukri.com" target="_blank">🔗 Naukri</a><br>
            <a href="https://indeed.com" target="_blank">🔗 Indeed</a>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="resource-card">
            <h4>🎓 Learning</h4>
            <a href="https://coursera.org" target="_blank">🔗 Coursera</a><br>
            <a href="https://skillsbuild.org" target="_blank">🔗 IBM SkillsBuild</a><br>
            <a href="https://udemy.com" target="_blank">🔗 Udemy</a>
        </div>
        """, unsafe_allow_html=True)
