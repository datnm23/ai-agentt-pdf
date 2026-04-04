from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

API_KEY = input("Nhập API key thứ 2: ").strip()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=API_KEY,
    temperature=0.1,
    max_output_tokens=256,
    thinking_budget=0,
)

try:
    resp = llm.invoke([HumanMessage(content="Trả lời ngắn: 2 + 2 = ?")])
    print("✅ OK:", resp.content[:100])
except Exception as e:
    print("❌ Lỗi:", e)
